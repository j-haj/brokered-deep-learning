package nameserver

import (
	"context"
	"flag"
	"sync"

	"google.golang.org/grpc"

	pbNS "github.com/j-haj/bdl/nameservice"
	log "github.com/sirupsen/logrus"
)

var (
	nameserverAddress = flag.String("nameserver_address", "localhost:10000",
		"Address used by the nameserver.")
	debug = flag.Bool("debug", false, "Enable debug logging.")
	logFile = flag.String("log_file", "", "Path to file used for logging.")
	timeout = flat.Int("timeout", 5, "Default timeout in seconds for RPCs.")
)


type broker struct {
	address string
	client pbNS.BrokerNameServiceClient
}

type brokerID int64

type nameserver struct {

	// Stores when the last heartbeat was received for each 
	heartbeats map[brokerID]float64
	
	// Stores the brokers known by the nameserver
	brokers map[brokerID]broker

	nextBrokerID brokerID

	mu sync.Mutex
}

func (ns *nameserver) Register(ctx context.Context, req *pbNS.RegistrationRequest) (pbNS.RegistrationResponse, error) {
	log.WithFields(log.Fields{
		"address": req.GetAddress(),
		"types": req.GetTypes(),
	}).Debug("Recevied broker registration request.")
	ns.mu.Lock()
	defer ns.mu.Unlock()

	// Create connection
	conn, err := grpc.Dial(req.GetAddress(), grpc.WithInsecure())
	if err != nil {
		log.WithFields(log.Fields{
			"address": req.GetAddress(),
		}).Error("Failed to establish connection.")
	}
	// Create broker nameservice client
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
