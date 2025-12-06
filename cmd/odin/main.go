package main

import (
	"context"
	"log"
	"net"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	"stan-cluster/internal/registry"
	pb "stan-cluster/proto"
)

type server struct {
	pb.UnimplementedSystemServer
	store *registry.Store
	// In a real system, we'd have a job queue channel here
	jobQueue chan *pb.Job
}

func (s *server) RegisterWorker(ctx context.Context, req *pb.RegisterRequest) (*pb.RegisterResponse, error) {
	workerID := uuid.New().String()
	log.Printf("Registering worker: %s (%s)", req.Hostname, workerID)

	s.store.AddWorker(&registry.Worker{
		ID:            workerID,
		Hostname:      req.Hostname,
		Tags:          req.Tags,
		LastHeartbeat: time.Now(),
		Status:        registry.StatusIdle,
	})

	return &pb.RegisterResponse{
		WorkerId: workerID,
		Success:  true,
	}, nil
}

func (s *server) Heartbeat(ctx context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	// log.Printf("Heartbeat from %s", req.WorkerId)
	s.store.UpdateWorkerHeartbeat(req.WorkerId, req.RunningJobs)
	return &pb.HeartbeatResponse{Acknowledge: true}, nil
}

func (s *server) GetJob(req *pb.GetJobRequest, stream pb.System_GetJobServer) error {
	log.Printf("Worker %s requested work stream", req.WorkerId)
	// Simple infinite loop to keep stream open and push jobs when available
	// In production, use a condition variable or channel per worker
	for {
		select {
		case job := <-s.jobQueue:
			log.Printf("Dispatching job %s to worker %s", job.JobId, req.WorkerId)
			if err := stream.Send(job); err != nil {
				return err
			}
		case <-time.After(1 * time.Second):
			// Keepalive / check if stream is dead
			// Check if worker is still connected via context?
			if stream.Context().Err() != nil {
				return stream.Context().Err()
			}
		}
	}
}

func (s *server) UpdateJobStatus(ctx context.Context, req *pb.JobUpdate) (*pb.JobUpdateResponse, error) {
	log.Printf("Job %s update from %s: %s", req.JobId, req.WorkerId, req.Status)
	if req.Status == "COMPLETED" {
		log.Printf("Job Result: %s", req.Stdout)
	}
	return &pb.JobUpdateResponse{Acknowledge: true}, nil
}

// Simple API to trigger a job for demo purposes
func (s *server) SubmitDemoJob() {
	// Wait a bit for workers to join
	time.Sleep(10 * time.Second)
	log.Println("Submitting demo job...")

	s.jobQueue <- &pb.Job{
		JobId:     uuid.New().String(),
		BinaryUrl: "echo",
		Args:      []string{"Hello", "Distributed", "World"},
	}
}

func main() {
	log.Println("Starting ODIN Master Node...")

	store := registry.NewStore()
	jobQ := make(chan *pb.Job, 100)

	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()
	srv := &server{store: store, jobQueue: jobQ}
	pb.RegisterSystemServer(s, srv)
	reflection.Register(s)

	// Start background reaper
	go func() {
		for {
			time.Sleep(10 * time.Second)
			store.ReapDeadWorkers(30 * time.Second)
		}
	}()

	// Start demo job submitter
	go srv.SubmitDemoJob()

	log.Printf("ODIN listening on :50051")
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
