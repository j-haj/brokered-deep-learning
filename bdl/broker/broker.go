package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"
	

	"google.golang.org/grpc"

	pbNS "github.com/j-haj/bdl/nameservice"
	pbHB "github.com/j-haj/bdl/heartbeat"
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
	connectionTimeout = flag.Int64("heartbeat_frequency", 100,
		"Time to wait in between heartbeats in seconds.")
	location = flag.String("location", "unknown", "Location of broker.")
)

type broker struct {
	brokerId int64
	nsClient pbNS.BrokerNameServiceClient
	hbClient pbHB.HeartbeatClient
	types []string
}

func NewBroker(types []string) (*broker, error) {
	conn, err := grpc.Dial(*nameserverAddress, grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("failed to establish connection with nameserver - %v", err)
	}
	
	b := &broker{
		brokerId: 0,
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

func (b *broker) sendHeartbeat() {
	for _ = range time.Tick(time.Duration(*connectionTimeout) * time.Second) {
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
