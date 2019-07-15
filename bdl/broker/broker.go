package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"
	

	"google.golang.org/grpc"

	log "github.com/sirupsen/logrus"
	
	pbBroker "github.com/j-haj/bdl/broker_comm"
	pbHB "github.com/j-haj/bdl/heartbeat"
	pbNS "github.com/j-haj/bdl/nameservice"
	pbResult "github.com/j-haj/bdl/result"
	pbTask "github.com/j-haj/bdl/task"
	taskQ "github.com/j-haj/bdl/task_queue"

)

var (
	brokerAddress = flag.String("broker_address", "localhost:10001",
		"Address used by broker.")
	nameserverAddress = flag.String("nameserver_address", "localhost:10000",
		"Address of the nameserver.")
	debug = flag.Bool("debug", false, "Enable debug logging.")
	logFile = flag.String("log_file", "", "Path to file used for logging.")
	timeout = flag.Int64("rpc_timeout", 1, "Timeout in seconds used for RPCs.")
	connectionTimeout = flag.Float64("heartbeat_frequency", 100.0,
		"Time to wait in between heartbeats in seconds.")
	location = flag.String("location", "unknown", "Location of broker.")
)

// taskID is a combination of task source and task ID  - <task source>:<task id>.
type taskID string
type brokerID int64

type task struct {
	id taskID
	owner brokerID
	taskProto *pbTask.Task
}

func newTask(owner brokerID, t *pbTask.Task) task {
	return task{buildTaskID(t), owner, t}
}

type brokerConnection struct {
	hbClient pbHB.HeartbeatClient
	status bool
}

type broker struct {
	// Timeout buffer is the fraction of the timeout that is used to send heartbeats.
	// For example, a timeoutBuffer of .75 means instead of sending a heartbeat message
	// every 100s the broker sends the heartbeat every 75s.
	timeoutBuffer float64
	brokerId brokerID
	nsClient pbNS.BrokerNameServiceClient
	hbClient pbHB.HeartbeatClient
	linkedBrokers map[brokerID]brokerConnection
	types []string
	heartbeats map[brokerID]time.Time
	// associatedTasks is a map from brokerID to taskID. This is used for dqueueing and
	// requeueing tasks.
	associatedTasks map[brokerID][]taskID
	// ownedTasks are tasks sent to this broker from a model
	ownedTasks map[taskID]task
	// queuedTasks are tasks waiting to be sent to an available worker
	queuedTasks taskQ.TaskQueue
	// borrowedTasks are tasks that have been sent from another broker
	borrowedTasks map[taskID]task
	// sharedTasks are tasks owned by this broker but sent to another broker
	sharedTasks map[taskID]brokerID
	// processingTasks are tasks currently being processed by a worker. If a task comes back
	// and is not in the processingTasks map the result should be dropped.
	processingTasks map[taskID]task
}

func NewBroker(types []string) (*broker, error) {
	conn, err := grpc.Dial(*nameserverAddress, grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("failed to establish connection with nameserver - %v", err)
	}
	
	b := &broker{
		timeoutBuffer: .75,
		brokerId: brokerID(0),
		nsClient: pbNS.NewBrokerNameServiceClient(conn),
		hbClient: pbHB.NewHeartbeatClient(conn),
		types: types,
	}

	err = b.registerWithNameserver()
	if err != nil {
		return nil, err
	}

	return b, nil
}

func (b *broker) registerWithNameserver() error {
	log.WithFields(log.Fields{
		"nameserver": nameserverAddress,
	}).Debug("Registering with nameserver.")

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeout) * time.Second)
	defer cancel()
	resp, err := b.nsClient.Register(ctx,
		&pbNS.RegistrationRequest{
			Address: *brokerAddress,
			Location: *location,
			Types: b.types,
		})
	if err != nil {
		return err
	}
	log.Debugf("Registration successful. Got ID: %d\n", resp.GetId())
	b.brokerId = resp.GetId()
	return nil
}

// sendHeartbeat sends heartbeat to nameserver and any connected brokers.
func (b *broker) sendHeartbeat() {
	for _ = range time.Tick(time.Duration(*connectionTimeout * b.timeoutBuffer) * time.Second) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeout) * time.Second)
		defer cancel()
		log.WithFields(log.Fields{
			"broker": b.brokerId,
			"address": *brokerAddress,
		}).Debug("Sending heartbeat.")
		resp, err := b.hbClient.Heartbeat(ctx, &pbHB.HeartbeatRequest{
			Id: b.brokerId,
			Address: *brokerAddress,
		})
		if err != nil {
			log.WithFields(log.Fields{
				"broker": b.brokerId,
				"nameserver": *nameserverAddress,
				"error": err,
			}).Error("Encountered error with nameserver.")
		}

		if resp.GetReregister() {
			log.WithFields(log.Fields{
				"broker": b.brokerId,
				"nameserver": *nameserverAddress,
			}).Debug("Received re-register request.")
			err = b.registerWithNameserver()
			if err != nil {
				log.WithFields(log.Fields{
					"broker": b.brokerId,
					"nameserver": *nameserverAddress,
					"error": err,
				}).Error("Failed to re-register with nameserver.")
			}
		}
	}
}

func (b *broker) checkHeartbeats() {
	for _ = range time.Tick(time.Duration(*connectionTimeout) * time.Second) {
		// Get timed out connections
		deadConnection := []brokerID{}
		for id, t := range b.heartbeats {
			if time.Since(t).Seconds() > *connectionTimeout {
				// Disconnect
				deadConnections = append(deadConnections, id)
			}
		}
		
		// Handle timed out connections
		// 1. Remove from heartbeats
		// 2. Requeue all shared tasks
		// 3. Drop all tasks enqueued and shared from lost broker
		for _, id := range deadConnections {
			// 1. Remove heartbeat
			delete(b.heartbeats, id)
			if tid, ok := b.associatedTasks[id]; !ok {
				// If there are no associated tasks with this broker
				// we are done.
				continue
			}
			tid := b.associatedTasks[id]
			if t, ok := b.sharedTasks[tid]; ok {
				// Shared tasks need to be requeued
			} else if _, ok := b.borrowedTasks[tid]; ok {
				// Borrowed tasks can be dropped since the broker will
				// re-send its tasks after a crash.
				delete(b.borrowedTasks, tid)
			}

			if _, ok := b.processingTasks[tid]; ok {
				// Remove task from processing task queue so its result will
				// be dropped when returned from worker.
				delete(b.processingTasks, tid)
			}
			
		}

	}
}

func (b *broker) SendAvailability(ctx context.Context, req *pbBroker.AvailabilityInfo) (*pbBroker.AvailabilityResponse, error) {
	return nil, fmt.Error("Not implemented")
}

func (b *broker) Connect(ctx context.Context, req *pbBroker.ConnectionRequest) (*pbBroker.ConnectionResponse, error) {
	return nil, fmt.Error("Not implemented")
}

func (b *broker) Disconnect(ctx context.Context, req *pbBroker.DisconnectRequest) (*pbBroker.DisconnectResponse, error) {
	return nil, fmt.Error("Not implemented")
}

// ShareTask is called when the broker is receiving a task from a linked broker. The linked
// broker is sending a task to this broker to share a task.
func (b *broker) ShareTask(ctx context.Context, req *pbBroker.ShareRequest) (*pbBroker.ShareResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	taskId := req.GetTask().GetTaskId()
	task := newTask(req.GetOriginatorId(), req.GetTask())
	b.borrowedTasks[taskId] = task
	b.associatedTasks[req.GetOriginatorId()] = taskId
	return nil, fmt.Error("Not implemented")
}

func (b *broker) ProcessResult(ctx context.Context, req *pbResult.Result) (*pbBroker.ProcessResponse, error) {
	return nil, fmt.Error("Not implemented")
}

func main() {
	flag.Parse()
	if *debug {
		log.SetLevel(log.DebugLevel)
	}
	if *logFile != "" {
		file, err := os.OpenFile(*logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			log.WithFields(log.Fields{
				"path": *logFile,
			}).Error("Failed to open file for writing.")
		} else {
			log.SetOutput(file)
		}
	}

	// Create server and listen
	// 	lis, err := net.Listen("tcp", *brokerAddress)
	// 	if err != nil {
	// 		log.WithFields(log.Fields{
	// 			"address": *brokerAddress,
	// 		}).Fatal("Failed to listen.")
	// 	}
	// 	log.WithFields(log.Fields{
	// 		"address": *brokerAddress,
	// 	}).Info("Broker listening.")
	//
	types := []string{"cpu"}
	b, err := NewBroker(types)
	go b.sendHeartbeat()
	go b.checkHeartbeats()
	if err != nil {
		log.Fatalf("Failed to create broker - %v\n", err)
	}
	
	if err != nil {
		fmt.Printf("Failed to register - %v\n", err)
	}

	for _ = range time.Tick(1 * time.Second) {
		fmt.Println("Waiting")
	}
}
