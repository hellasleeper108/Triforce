package main

import (
	"context"
	"io"
	"log"
	"os"
	"os/exec"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "stan-cluster/proto"
)

const (
	masterAddr = "localhost:50051"
)

type Worker struct {
	client   pb.SystemClient
	workerID string
	conn     *grpc.ClientConn
}

func NewWorker() *Worker {
	conn, err := grpc.NewClient(masterAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	return &Worker{
		client: pb.NewSystemClient(conn),
		conn:   conn,
	}
}

func (w *Worker) Register() {
	hostname, _ := os.Hostname()
	log.Printf("Registering with master at %s...", masterAddr)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()

	r, err := w.client.RegisterWorker(ctx, &pb.RegisterRequest{
		Hostname: hostname,
		CpuCores: 4,               // TODO: Detect real stats
		RamMb:    16384,           // TODO: Detect real stats
		Tags:     []string{"gpu"}, // Demo tags
	})
	if err != nil {
		log.Fatalf("could not register: %v", err)
	}

	if !r.Success {
		log.Fatalf("registration denied")
	}

	w.workerID = r.WorkerId
	log.Printf("Registered successfully! Worker ID: %s", w.workerID)
}

func (w *Worker) StartHeartbeat() {
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		for range ticker.C {
			_, err := w.client.Heartbeat(context.Background(), &pb.HeartbeatRequest{
				WorkerId:    w.workerID,
				RunningJobs: 0, // Placeholder
				MemoryUsage: 0,
			})
			if err != nil {
				log.Printf("Heartbeat failed: %v", err)
			}
		}
	}()
}

func (w *Worker) RunJobLoop() {
	stream, err := w.client.GetJob(context.Background(), &pb.GetJobRequest{
		WorkerId: w.workerID,
	})
	if err != nil {
		log.Fatalf("failed to open job stream: %v", err)
	}

	log.Println("Waiting for jobs...")
	for {
		job, err := stream.Recv()
		if err == io.EOF {
			log.Println("Master closed stream")
			return
		}
		if err != nil {
			log.Fatalf("Stream error: %v", err)
		}

		w.executeJob(job)
	}
}

func (w *Worker) executeJob(job *pb.Job) {
	log.Printf("Received Job %s: %s %v", job.JobId, job.BinaryUrl, job.Args)

	// Report START
	w.client.UpdateJobStatus(context.Background(), &pb.JobUpdate{
		JobId:    job.JobId,
		WorkerId: w.workerID,
		Status:   "RUNNING",
	})

	// Execute
	cmd := exec.Command(job.BinaryUrl, job.Args...)
	out, err := cmd.CombinedOutput()

	exitCode := 0
	status := "COMPLETED"
	if err != nil {
		log.Printf("Job failed: %v", err)
		exitCode = 1
		status = "FAILED"
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		}
	}

	log.Printf("Job Finished. Output: %s", string(out))

	// Report FINISH
	w.client.UpdateJobStatus(context.Background(), &pb.JobUpdate{
		JobId:    job.JobId,
		WorkerId: w.workerID,
		Status:   status,
		ExitCode: int32(exitCode),
		Stdout:   string(out),
		Stderr:   "",
	})
}

func main() {
	w := NewWorker()
	defer w.conn.Close()

	w.Register()
	w.StartHeartbeat()
	w.RunJobLoop()
}
