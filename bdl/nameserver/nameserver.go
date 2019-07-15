package main

import (
	"context"
	"flag"
	"math/rand"
	"net"
	"os"
	"sync"
	"time"

	"google.golang.org/grpc"

	pbNS "github.com/j-haj/bdl/nameservice"
	pbHB "github.com/j-haj/bdl/heartbeat"
	log "github.com/sirupsen/logrus"
)

var (
	nameserverAddress = flag.String("nameserver_address", "localhost:10000", "Address used by the nameserver.")
	debug             = flag.Bool("debug", false, "Enable debug logging.")
	logFile           = flag.String("log_file", "", "Path to file used for logging.")
	timeout           = flag.Int("timeout", 5, "Default timeout in seconds for RPCs.")
	connectionTimeout = flag.Float64("connection_timeout", 120, "Time to wait in seconds befor removing a broker.")
	hbCheckFrequency = flag.Int64("heartbeat_check_freq", 60, "Time in seconds between checking which broker connections have timed out.")
)

type broker struct {
	address  string
	location string
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

// NewNameServer constructs a new nameserver.
func NewNameServer() *nameserver {
	return &nameserver{heartbeats: make(map[brokerID]time.Time),
		brokers: make(map[brokerID]*broker),
		locations: make(map[string]map[brokerID]bool),
		nextBrokerID: brokerID(0),
	}
		
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

	// Create broker nameservice client
	id := ns.nextBrokerID
	ns.brokers[id] = &broker{address, location}
	ns.heartbeats[id] = time.Now()
	if ns.locations[location] == nil {
		ns.locations[location] = make(map[brokerID]bool)
	}
	ns.locations[location][id] = true
	ns.nextBrokerID++
	return &pbNS.RegistrationResponse{Id: int64(id)}, nil
}

// Receive a heartbeat from a broker.
func (ns *nameserver) Heartbeat(ctx context.Context, req *pbHB.HeartbeatRequest) (*pbHB.HeartbeatResponse, error) {
	rcvd := time.Now()
	id := brokerID(req.GetId())
	log.WithFields(log.Fields{
		"id":   id,
		"time": rcvd,
	}).Debug("Received heartbeat.")
	ns.mu.Lock()
	defer ns.mu.Unlock()
	if _, ok := ns.heartbeats[id]; !ok {
		// Handle the case where we don't have a heartbeat record - this occurs when a broker
		// has been dropped
		log.WithFields(log.Fields{
			"broker_id": id,
			"nameserver": *nameserverAddress,
		}).Debug("Unrecognized broker.")
		return &pbHB.HeartbeatResponse{Reregister: true}, nil
	}
	ns.heartbeats[id] = rcvd
	return &pbHB.HeartbeatResponse{Reregister: false}, nil
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
	for _ = range time.Tick(time.Duration(*hbCheckFrequency)*time.Second) {
		ns.mu.Lock()
		dead := []brokerID{}
		// Find dead connections
		for k, v := range ns.heartbeats {
			if time.Since(v).Seconds() > *connectionTimeout {
				dead = append(dead, k)
			}
		}

		// Remove dead connections
		for _, k := range dead {
			l := ns.brokers[k].location
			if _, ok := ns.heartbeats[k]; ok {
				delete(ns.heartbeats, k)
			}
			if _, ok := ns.brokers[k]; ok {
				delete(ns.brokers, k)
			}
			if _, ok := ns.locations[l][k]; ok {
				delete(ns.locations[l], k)
			}
		}
		ns.mu.Unlock()
	}
}

func main() {
	flag.Parse()
	if *debug {
		log.SetLevel(log.DebugLevel)
		log.Debug("Using DEBUG logging.")
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
	lis, err := net.Listen("tcp", *nameserverAddress)
	if err != nil {
		log.WithFields(log.Fields{
			"address": *nameserverAddress,
		}).Fatal("Failed to listen.")
	}
	log.WithFields(log.Fields{
		"address": *nameserverAddress,
	}).Info("Nameserver listening.")
	s := grpc.NewServer()
	n := NewNameServer()

	go n.checkHeartbeats()
	pbNS.RegisterBrokerNameServiceServer(s, n)
	pbHB.RegisterHeartbeatServer(s, n)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("Failed to server - %v", err)
	}
}
