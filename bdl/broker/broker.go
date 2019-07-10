package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"
	

	"google.golang.org/grpc"

	pbNS "github.com/j-haj/bdl/nameservice"
	log "github.com/sirupsen/logrus"
)

var (
	brokerAddress = flag.String("broker_address", "localhost:10001",
		"Address used by broker.")
	nameserverAddress = flag.String("nameserver_address", "localhost:10000",
		"Address of the nameserver.")
	debug = flag.Bool("debug", false, "Enable debug logging.")
	logFile = flag.String("log_file", "", "Path to file used for logging.")
	timeout = flag.Int64("rpc_timeout", 1, "Timeout in seconds used for RPCs.")
)

type broker struct {
	nsClient pbNS.BrokerNameServiceClient
}

func NewBroker() (*broker, error) {
	conn, err := grpc.Dial(*nameserverAddress, grpc.WithInsecure())
	if err != nil {
		log.WithFields(log.Fields{
			"address": *nameserverAddress,
		}).Error("Failed to establish connection with nameserver.")
		return nil, fmt.Errorf("failed to establish connection with nameserver - %v", err)
	}
	
	return &broker{nsClient: pbNS.NewBrokerNameServiceClient(conn)}, nil
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
	b, err := NewBroker()
	if err != nil {
		log.Fatalf("Failed to create broker - %v\n", err)
	}

	ctx, _ := context.WithDeadline(context.Background(), time.Now().Add(time.Duration(*timeout) * time.Second))
	
	resp, err := b.nsClient.Register(ctx, &pbNS.RegistrationRequest{Address: "localhost:10001",
		Location: "iowa"})
	if err != nil {
		fmt.Printf("Failed to register - %v\n", err)
	}
	fmt.Printf("Received ID: %d\n", resp.GetId())
}
