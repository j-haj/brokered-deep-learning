package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"

	log "github.com/sirupsen/logrus"

	pbBroker "github.com/j-haj/bdl/broker_comm"
	pbHB "github.com/j-haj/bdl/heartbeat"
	pbMS "github.com/j-haj/bdl/model_service"
	pbNS "github.com/j-haj/bdl/nameservice"
	pbResult "github.com/j-haj/bdl/result"
	task "github.com/j-haj/bdl/task"
	taskQ "github.com/j-haj/bdl/task_queue"
	pbTask "github.com/j-haj/bdl/task_service"
)

var (
	brokerAddress = flag.String("broker_address", "localhost:10001",
		"Address used by broker.")
	nameserverAddress = flag.String("nameserver_address", "localhost:10000",
		"Address of the nameserver.")
	debug             = flag.Bool("debug", false, "Enable debug logging.")
	logFile           = flag.String("log_file", "", "Path to file used for logging.")
	timeout           = flag.Int64("rpc_timeout", 1, "Timeout in seconds used for RPCs.")
	connectionTimeout = flag.Float64("heartbeat_frequency", 100.0,
		"Time to wait in between heartbeats in seconds.")
	location              = flag.String("location", "unknown", "Location of broker.")
	nAttemptedConnections = flag.Int("n_attempted_connections", 1,
		"The number of brokers this broker will attempt to connect to.")
)

// taskID is a combination of task source and task ID  - <task source>:<task id>.
type brokerID string
type modelAddress string

type ownedTask struct {
	owner    brokerID
	taskData task.Task
}

func newTask(owner brokerID, t *pbTask.Task) ownedTask {
	return ownedTask{owner, task.Task{task.TaskID(t.GetTaskId()), t}}
}

type brokerConnection struct {
	hbClient     pbHB.HeartbeatClient
	brokerClient pbBroker.InterBrokerCommClient
	isAvailable  bool
}

func newBrokerConnection(address string) (brokerConnection, error) {
	conn, err := grpc.Dial(address, grpc.WithInsecure())
	if err != nil {
		return brokerConnection{}, fmt.Errorf("failed to establish broker connection - %v", err)
	}
	brokerClient := pbBroker.NewInterBrokerCommClient(conn)
	hbClient := pbHB.NewHeartbeatClient(conn)

	return brokerConnection{
		hbClient:     hbClient,
		brokerClient: brokerClient,
		isAvailable:  true,
	}, nil
}

type modelClient struct {
	client pbResult.ResultServiceClient
	address modelAddress
}

func newModelClient(address modelAddress) (modelClient, error) {
	conn, err := grpc.Dial(string(address), grpc.WithInsecure())
	if err != nil {
		log.WithFields(log.Fields{
			"model_address": address,
		}).Error("Failed to create model client.")
		return modelClient{}, fmt.Errorf("failed to create model client connection - %v", err)
	}
	log.WithFields(log.Fields{
		"model_address": address,
	}).Debug("Established dialed RPC connection for model client.")
	client := pbResult.NewResultServiceClient(conn)
	if client == nil {
		log.Error("NIL CLIENT")
	}
	return modelClient{client, address}, nil
}

type broker struct {
	// Timeout buffer is the fraction of the timeout that is used to send heartbeats.
	// For example, a timeoutBuffer of .75 means instead of sending a heartbeat message
	// every 100s the broker sends the heartbeat every 75s.
	timeoutBuffer float64
	// brokerID is the ID given to this broker from the nameserver
	brokerId brokerID
	// nsClient handles interactions with the nameserver
	nsClient pbNS.BrokerNameServiceClient
	// hbClient handles heartbeats with the nameserver
	hbClient pbHB.HeartbeatClient
	// modelClients store the clients to send results back to known models
	modelClients map[modelAddress]modelClient
	// linkedBrokers are brokers that have an established link with this broker
	linkedBrokers map[brokerID]brokerConnection
	// types are the computation types this broker is capable of handling
	types []string
	// heartbeats tracks the received heartbeats from other linked brokers
	heartbeats map[brokerID]time.Time
	// associatedTasks is a map from brokerID to taskID. This is used for dqueueing and
	// requeueing tasks.
	associatedTasks map[brokerID][]task.TaskID
	// ownedTasks are tasks sent to this broker from a model
	ownedTasks map[task.TaskID]ownedTask
	// queuedTasks are tasks waiting to be sent to an available worker
	queuedTasks taskQ.TaskQueue
	// borrowedTasks are tasks that have been sent from another broker
	borrowedTasks map[task.TaskID]ownedTask
	// sharedTasks are tasks owned by this broker but sent to another broker
	sharedTasks map[task.TaskID]brokerID
	// processingTasks are tasks currently being processed by a worker. If a task comes back
	// and is not in the processingTasks map the result should be dropped.
	processingTasks map[task.TaskID]ownedTask
	// maxBorrowedCapacity is the number of shared tasks the broker is willing to accept
	maxBorrowedCapacity int

	attemptingConnections bool
	mu sync.Mutex
}

// NewBroker creates a new broker with a maximum borrow capactiy and allowed computation types.
func NewBroker(maxBorrowedCapacity int, types []string) (*broker, error) {
	conn, err := grpc.Dial(*nameserverAddress, grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("failed to establish connection with nameserver - %v", err)
	}

	b := &broker{
		timeoutBuffer:       .75,
		brokerId:            brokerID(0),
		nsClient:            pbNS.NewBrokerNameServiceClient(conn),
		hbClient:            pbHB.NewHeartbeatClient(conn),
		types:               types,
		maxBorrowedCapacity: maxBorrowedCapacity,
		modelClients: make(map[modelAddress]modelClient),
		linkedBrokers: make(map[brokerID]brokerConnection),
		heartbeats: make(map[brokerID]time.Time),
		associatedTasks: make(map[brokerID][]task.TaskID),
		ownedTasks: make(map[task.TaskID]ownedTask),
		borrowedTasks: make(map[task.TaskID]ownedTask),
		sharedTasks: make(map[task.TaskID]brokerID),
		processingTasks: make(map[task.TaskID]ownedTask),
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

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeout)*time.Second)
	defer cancel()
	resp, err := b.nsClient.Register(ctx,
		&pbNS.RegistrationRequest{
			Address:  *brokerAddress,
			Location: *location,
			Types:    b.types,
		})
	if err != nil {
		return err
	}
	log.Debugf("Registration successful. Got ID: %s\n", resp.GetId())
	b.brokerId = brokerID(resp.GetId())
	return nil
}

// sendHeartbeat sends heartbeat to nameserver and any connected brokers.
func (b *broker) sendHeartbeat() {
	for _ = range time.Tick(time.Duration(*connectionTimeout*b.timeoutBuffer) * time.Second) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeout)*time.Second)
		defer cancel()
		log.WithFields(log.Fields{
			"broker":  b.brokerId,
			"address": *brokerAddress,
		}).Debug("Sending heartbeat.")
		resp, err := b.hbClient.Heartbeat(ctx, &pbHB.HeartbeatRequest{
			Id:      string(b.brokerId),
			Address: *brokerAddress,
		})
		if err != nil {
			log.WithFields(log.Fields{
				"broker":     b.brokerId,
				"nameserver": *nameserverAddress,
				"error":      err,
			}).Error("Encountered error with nameserver.")
		}

		if resp.GetReregister() {
			log.WithFields(log.Fields{
				"broker":     b.brokerId,
				"nameserver": *nameserverAddress,
			}).Debug("Received re-register request.")
			err = b.registerWithNameserver()
			if err != nil {
				log.WithFields(log.Fields{
					"broker":     b.brokerId,
					"nameserver": *nameserverAddress,
					"error":      err,
				}).Error("Failed to re-register with nameserver.")
			}
		}
	}
}

func (b *broker) checkHeartbeats() {
	for _ = range time.Tick(time.Duration(*connectionTimeout) * time.Second) {
		// Get timed out connections
		deadConnections := []brokerID{}
		for id, t := range b.heartbeats {
			if time.Since(t).Seconds() > *connectionTimeout {
				// Disconnect
				deadConnections = append(deadConnections, id)
			}
		}

		// Handle timed out connections
		// 1. Remove broker from linked brokers
		// 2. Remove from heartbeats
		// 3. Remove associated tasks
		// 4. Remove associated tasks from queue tasks
		// 5. Remove borrowed tasks
		// 6. Remove from shared task and put the the front of the task queue
		// 7. Remove from processing tracking
		b.mu.Lock()
		for _, id := range deadConnections {
			// 1. Remove heartbeat
			delete(b.heartbeats, id)

			// 2. Remove linked broker entry
			delete(b.linkedBrokers, id)

			// 3. Get associated task IDs
			if _, ok := b.associatedTasks[id]; !ok {
				// If there are no associated tasks with this broker
				// we are done.
				continue
			}
			tids := b.associatedTasks[id]
			for _, tid := range tids {
				// Remove from task queue
				b.queuedTasks.Remove(tid)

				if _, ok := b.borrowedTasks[tid]; ok {
					delete(b.borrowedTasks, tid)
				} else if _, ok := b.sharedTasks[tid]; ok {
					t := b.processingTasks[tid]
					b.queuedTasks.PushFront(t.taskData.TaskProto)
					delete(b.sharedTasks, tid)
				}
				if _, ok := b.processingTasks[tid]; ok {
					delete(b.processingTasks, tid)
				}

			}
			if _, ok := b.associatedTasks[id]; ok {
				delete(b.associatedTasks, id)
			}
		}
		b.mu.Unlock()
	}
}

func (b *broker) Heartbeat(ctx context.Context, req *pbHB.HeartbeatRequest) (*pbHB.HeartbeatResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	id := brokerID(fmt.Sprintf("%s#%d", req.GetAddress(), req.GetId()))
	reregister := true
	if _, ok := b.heartbeats[id]; ok {
		reregister = false
		b.heartbeats[id] = time.Now()
	}

	return &pbHB.HeartbeatResponse{Reregister: reregister}, nil
}

func (b *broker) SendAvailability(ctx context.Context, req *pbBroker.AvailabilityInfo) (*pbBroker.AvailabilityResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Construct ID
	id := brokerID(req.GetBrokerId())

	// Update availability
	if broker, ok := b.linkedBrokers[id]; ok {
		broker.isAvailable = req.GetAvailable()
		return &pbBroker.AvailabilityResponse{}, nil
	}
	return nil, errors.New("broker not recognized")
}

// Connect is called by a broker to establish a link with another broker.
func (b *broker) Connect(ctx context.Context, req *pbBroker.ConnectionRequest) (*pbBroker.ConnectionResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	id := brokerID(req.GetBrokerId())
	if _, ok := b.linkedBrokers[id]; ok {
		log.Debugf("Already connected with broker %s", id)
		return &pbBroker.ConnectionResponse{}, nil
	}
	address := strings.Split(req.GetBrokerId(), "#")[0]

	c, err := newBrokerConnection(address)
	if err != nil {
		return nil, err
	}
	b.linkedBrokers[id] = c
	b.heartbeats[id] = time.Now()

	return &pbBroker.ConnectionResponse{}, nil
}

// Disconnect is called by a broker to destroy a link with another broker.
func (b *broker) Disconnect(ctx context.Context, req *pbBroker.DisconnectRequest) (*pbBroker.DisconnectResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	id := brokerID(req.GetBrokerId())

	// Remove broker as a linked broker
	if _, ok := b.linkedBrokers[id]; ok {
		delete(b.linkedBrokers, id)
	}

	// Remove from heartbeat list
	if _, ok := b.heartbeats[id]; ok {
		delete(b.heartbeats, id)
	}

	// Remove any associated tasks
	if _, ok := b.associatedTasks[id]; ok {
		for _, tid := range b.associatedTasks[id] {
			if _, ok := b.processingTasks[tid]; ok {
				delete(b.processingTasks, tid)
			}
		}
		delete(b.associatedTasks, id)
	}

	if len(b.linkedBrokers) < *nAttemptedConnections {
		go b.attemptConnections()
	}
	return nil, errors.New("Not implemented")
}

// ShareTask is called when the broker is receiving a task from a linked broker. The linked
// broker is sending a task to this broker to share a task.
func (b *broker) ShareTask(ctx context.Context, req *pbBroker.ShareRequest) (*pbBroker.ShareResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	taskId := task.TaskID(req.GetTaskProto().GetTaskId())
	bid := brokerID(req.GetOriginatorId())
	t := newTask(bid, req.GetTaskProto())
	b.borrowedTasks[taskId] = t
	b.associatedTasks[bid] = append(b.associatedTasks[bid], taskId)
	return &pbBroker.ShareResponse{Ok: true}, nil
}

// SendResult is called by a worker to report the result back to the broker.
func (b *broker) SendResult(ctx context.Context, req *pbResult.Result) (*pbResult.ResultResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()


	tid := task.TaskID(req.GetTaskId())
	destination := req.GetDestination()
	log.WithFields(log.Fields{
		"destination": destination,
		"task_id": tid,
	}).Debug("Received result from worker.")
	if _, ok := b.processingTasks[tid]; !ok {
		// If the task is not in processing tasks then we should
		// drop the result
		log.WithFields(log.Fields{
			"task_id": tid,
			"destination": req.GetDestination(),
		}).Debug("Dropping result")
		return &pbResult.ResultResponse{}, fmt.Errorf("dropped result")
	} else if _, ok := b.ownedTasks[tid]; ok {
		// This is an owned task
		c, cancel := context.WithTimeout(context.Background(),
			time.Duration(*timeout) * time.Second)
		defer cancel()
		mc := b.modelClients[modelAddress(destination)]
		_, err := mc.client.SendResult(c, req)
		log.WithFields(log.Fields{
			"task_id": tid,
			"destination": destination,
		}).Debug("Sent result to model.")
		if err != nil {
			log.Errorf("non-nil error after sending result to model - %v", err)
		}
		return &pbResult.ResultResponse{}, nil
	} else if t, ok := b.borrowedTasks[tid]; ok {
		c, cancel := context.WithTimeout(context.Background(),
			time.Duration(*timeout) * time.Second)
		defer cancel()
		b := b.linkedBrokers[t.owner]
		b.brokerClient.ProcessResult(c, req)
		log.WithFields(log.Fields{
			"task_id": tid,
			"destination": destination,
		}).Debug("Returning result to owning broker.")
		return &pbResult.ResultResponse{}, nil
	} else {
		// Unknown task
		log.WithFields(log.Fields{
			"task_id": tid,
			"destination": req.GetDestination(),
		}).Error("Result received for unknown task. Dropping")
	}
	return &pbResult.ResultResponse{}, fmt.Errorf("unknown result received at %s", b.brokerId)
}

// ProcessResult is used to send a shared task back to the owning broker.
func (b *broker) ProcessResult(ctx context.Context, req *pbResult.Result) (*pbBroker.ProcessResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Check if this is a known task
	tid := task.TaskID(req.GetTaskId())
	if _, ok := b.ownedTasks[tid]; ok {
		// This is a known task
		log.WithFields(log.Fields{
			"task_id":     req.GetTaskId(),
			"destination": req.GetDestination(),
		}).Debug("Processing known task.")

		// Get model client and send result
		destination := modelAddress(req.GetDestination())
		if mc, ok := b.modelClients[destination]; ok {
			ctx2, cancel := context.WithTimeout(context.Background(),
				time.Duration(*timeout) * time.Second)
			defer cancel()
			
			mc.client.SendResult(ctx2, req)
			log.WithFields(log.Fields{
				"task_id": req.GetTaskId(),
				"model_destination": req.GetDestination(),
			}).Debug("Sent results to model.")
		} else {
			log.WithFields(log.Fields{
				"model_destination": req.GetDestination(),
				"task_id": req.GetTaskId(),
			}).Error("Failed to find model client.")
		}
		
		return &pbBroker.ProcessResponse{}, nil
	}
	log.WithFields(log.Fields{
		"task_id":     req.GetTaskId(),
		"destination": req.GetDestination(),
	}).Error("Unrecognized tasked received for processing.")
	return &pbBroker.ProcessResponse{}, nil
}

// RegisterModel handles the registration request of a model.
func (b *broker) RegisterModel(ctx context.Context, req *pbMS.RegistrationRequest) (*pbMS.RegistrationResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	log.WithFields(log.Fields{
		"model_address": req.GetAddress(),
	}).Debug("Received model registration request.")
	address := modelAddress(req.GetAddress())
	client, err := newModelClient(address)
	if err != nil {
		log.WithFields(log.Fields{
			"model_address": req.GetAddress(),
		}).Error("Failed to create model result client.")
		return &pbMS.RegistrationResponse{Success: false}, err
	}
	b.modelClients[address] = client
	return &pbMS.RegistrationResponse{Success: true}, nil
}

func (b *broker) SendTask(ctx context.Context, req *pbTask.Task) (*pbMS.SendResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	log.WithFields(log.Fields{
		"task_id": req.GetTaskId(),
		"source_address": req.GetSource(),
	}).Debug("Received task")
	
	// create an owned task
	t := newTask(b.brokerId, req)

	// add to ownedTasks
	b.ownedTasks[task.TaskID(req.GetTaskId())] = t

	// queue task for processing
	b.queuedTasks.Push(req)
	
	return &pbMS.SendResponse{}, nil
}

func (b *broker) RequestTask(ctx context.Context, req *pbTask.TaskRequest) (*pbTask.Task, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	t, err := b.queuedTasks.Pop()
	if err != nil {
		log.Debugf("Failed to pop from task queue - %s", err)
		return &pbTask.Task{TaskId: "", Source: ""}, err
	}
	log.Debugf("Received task request - sending task %s.", t.GetTaskId())	
	b.processingTasks[task.TaskID(t.GetTaskId())] = newTask(b.brokerId, t)
	return t, nil
}

// attemptConnections attempts to connect the broker with other brokers using an exponential backoff
// that resets after an hour.
func (b *broker) attemptConnections() {
	b.mu.Lock()
	b.attemptingConnections = true
	b.mu.Unlock()
	defer func() {
		log.Debug("Done attempting connections.")
		b.attemptingConnections = false
	}()
	
	backoff := 0.0
	count := 0
	for {
		if len(b.linkedBrokers) > *nAttemptedConnections {
			return
		}
		// Exponential backoff plus noise
		<-time.After(time.Duration(backoff) * time.Second)
		backoff = math.Max(1.0, math.Exp2(float64(count)) + rand.NormFloat64())
		log.Debugf("current backoff: %v", backoff)
		count++
		
		// Request broker from nameserver
		ctx, cancel1 := context.WithTimeout(context.Background(),
			time.Duration(*timeout) * time.Second)
		defer cancel1()
		resp, err := b.nsClient.RequestBroker(ctx,
			&pbNS.BrokerRequest{Location: "any", Address: *brokerAddress})
		if err != nil {
			log.Errorf("Failed to request broker - %v", err)
			continue
		} else if resp.GetAddress() == "" {
			log.Error("Received empty broker address from nameserver.")
			continue
		}

		// Attempt to connect
		brokerClient, err := newBrokerConnection(resp.GetAddress())
		if err != nil {
			log.WithFields(log.Fields{
				"remote_address": resp.GetAddress(),
				"error": err,
			}).Error("failed to establish RPC connection.")
			continue
		}
		ctx2, cancel2 := context.WithTimeout(context.Background(), time.Duration(*timeout) * time.Second)
		defer cancel2()
		connResp, err := brokerClient.brokerClient.Connect(ctx2, &pbBroker.ConnectionRequest{BrokerId: string(b.brokerId)})
		if err != nil {
			log.WithFields(log.Fields{
				"remote_address": resp.GetAddress(),
				"error": err,
			}).Error("connection request with remote broker failed.")
			continue
		}
		b.mu.Lock()
		b.linkedBrokers[brokerID(connResp.GetBrokerId())] = brokerClient
		b.mu.Unlock()
		log.WithFields(log.Fields{
			"remote_address": resp.GetAddress(),
			"local_address": *brokerAddress,
		}).Info("Successfully establish broker-broker connection.")
	}
}

// server contains the main-loop of the broker.
func (b *broker) serve() {
	for {
		if len(b.linkedBrokers) < *nAttemptedConnections && !b.attemptingConnections {
			go b.attemptConnections()
		}
		<-time.After(60 * time.Second)
	}
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
	lis, err := net.Listen("tcp", *brokerAddress)
	if err != nil {
		log.WithFields(log.Fields{
			"address": *brokerAddress,
		}).Fatal("Failed to listen.")
	}
	log.WithFields(log.Fields{
		"address": *brokerAddress,
	}).Info("Broker listening.")

	s := grpc.NewServer(grpc.MaxRecvMsgSize(500 * 1024 * 1024),
		grpc.MaxSendMsgSize(500 * 1024 * 1024))
	
	types := []string{"cpu"}
	b, err := NewBroker(10, types)
	go b.sendHeartbeat()
	go b.checkHeartbeats()
	pbHB.RegisterHeartbeatServer(s, b)
	pbMS.RegisterModelServiceServer(s, b)
	pbTask.RegisterTaskServiceServer(s, b)
	pbResult.RegisterResultServiceServer(s, b)
	pbBroker.RegisterInterBrokerCommServer(s, b)
	go b.serve()
	if err != nil {
		log.Fatalf("Failed to create broker - %v\n", err)
	}

	if err != nil {
		fmt.Printf("Failed to register - %v\n", err)
	}

	if err := s.Serve(lis); err != nil {
		log.Fatalf("Failed to server: %v", err)
	}
}
