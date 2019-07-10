package nameserver

import (
	"context"
	"flag"
	"math/rand"
	"sync"
	"time"

	"google.golang.org/grpc"

	pbNS "github.com/j-haj/bdl/nameservice"
	log "github.com/sirupsen/logrus"
)

var (
	nameserverAddress = flag.String("nameserver_address", "localhost:10000",
		"Address used by the nameserver.")
	debug             = flag.Bool("debug", false, "Enable debug logging.")
	logFile           = flag.String("log_file", "", "Path to file used for logging.")
	timeout           = flag.Int("timeout", 5, "Default timeout in seconds for RPCs.")
	connectionTimeout = flag.Float64("connection_timeout", 120, "Time to wait in seconds befor removing a broker.")
)

type broker struct {
	address  string
	location string
	client   pbNS.BrokerNameServiceClient
}

type brokerID int64

type nameserver struct {

	// Stores when the last heartbeat was received for each
	heartbeats map[brokerID]time.Time

	// Stores the brokers known by the nameserver
	brokers map[brokerID]*broker

	locations map[string]map[brokerID]bool

	nextBrokerID brokerID

	mu sync.Mutex
}

// Register a broker with the nameserver
func (ns *nameserver) Register(ctx context.Context, req *pbNS.RegistrationRequest) (*pbNS.RegistrationResponse, error) {
	address := req.GetAddress()
	location := req.GetLocation()
	types := req.GetTypes()
	log.WithFields(log.Fields{
		"address":  address,
		"location": location,
		"types":    types,
	}).Debug("Recevied broker registration request.")
	ns.mu.Lock()
	defer ns.mu.Unlock()

	// Create connection
	// TODO: I'm not sure we actually need this connection...
	conn, err := grpc.Dial(address, grpc.WithInsecure())
	if err != nil {
		log.WithFields(log.Fields{
			"address": address,
		}).Error("Failed to establish connection.")
		return nil, fmt.Errorf("failed to connect to %s - %v", req.GetAddress(), err)
	}

	// Create broker nameservice client
	id := ns.nextBrokerID
	ns.brokers[id] = &broker{address, location, req.NewBrokerServiceClient(conn)}
	if ns.locations[location] == nil {
		ns.locations[location] = make(map[brokerID]bool)
	}
	ns.locations[location][id] = true
	ns.nextBrokerID++
	return &pbNS.RegistrationResponse{Id: id}, nil
}

// Receive a heartbeat from a broker.
func (ns *nameserver) Heartbeat(ctx context.Context, req *pbNS.HeartbeatRequest) (*pbNS.HeartbeatResponse, error) {
	rcvd := time.Now()
	id := brokerID(req.GetId())
	log.WithFields(log.Fields{
		"id":   id,
		"time": rcvd,
	}).Debug("Received heartbeat.")
	ns.mu.Lock()
	defer ns.mu.Unlock()
	if p, ok := ns.heartbeats[id]; !ok {
		// Handle the case where we don't have a heartbeat record - this occurs when a broker
		// has been dropped
		return &pbNS.HeartbeatResponse{Reregister: true}, nil
	}
	ns.heartbeats[id] = rcvd
	return &pbNS.HeartbeatResponse{Reregister: false}, nil
}

func (ns *nameserver) RequestBroker(ctx context.Context, req *pbNS.BrokerRequest) (*pbNS.BrokerInfo, error) {
	ns.mu.Lock()
	defer ns.mu.Unlock()
	location := req.GetLocation()
	n := len(ns.locations[location])
	idx := brokerID(rand.Intn(n))
	return &pbNS.BrokerInfo{Address: ns.brokers[idx].address}, nil
}

func (ns *nameserver) checkHeartbeats() {
	ns.mu.Lock()
	defer ns.mu.Unlock()
	dead := &[]int{}
	// Find dead connections
	for k, v := range ns.heartbeats {
		if time.Since(v).Seconds() > connectionTimeout {
			dead = append(dead, k)
		}
	}

	// Remove dead connections
	for _, k := range dead {
		l := ns.brokers[k].location
		delete(ns.heartbeats, k)
		delete(ns.brokers, k)
		delete(ns.locations[l], k)
	}
}

func main() {
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
}
