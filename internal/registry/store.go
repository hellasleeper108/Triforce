package registry

import (
	"sync"
	"time"
)

type WorkerStatus string

const (
	StatusIdle    WorkerStatus = "IDLE"
	StatusBusy    WorkerStatus = "BUSY"
	StatusOffline WorkerStatus = "OFFLINE"
)

type Worker struct {
	ID            string
	Hostname      string
	Tags          []string
	LastHeartbeat time.Time
	Status        WorkerStatus
	RunningJobs   int32
	Address       string // Needed if we want to reverse-dial, though here we use pull
}

type JobStatus string

const (
	JobPending   JobStatus = "PENDING"
	JobAssigned  JobStatus = "ASSIGNED"
	JobRunning   JobStatus = "RUNNING"
	JobCompleted JobStatus = "COMPLETED"
	JobFailed    JobStatus = "FAILED"
)

type Job struct {
	ID          string
	Spec        interface{} // Placeholder for job details
	AssignedTo  string
	Status      JobStatus
	Result      string
	SubmittedAt time.Time
}

type Store struct {
	mu      sync.RWMutex
	Workers map[string]*Worker
	Jobs    map[string]*Job
}

func NewStore() *Store {
	return &Store{
		Workers: make(map[string]*Worker),
		Jobs:    make(map[string]*Job),
	}
}

func (s *Store) AddWorker(w *Worker) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Workers[w.ID] = w
}

func (s *Store) GetWorker(id string) (*Worker, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	w, ok := s.Workers[id]
	return w, ok
}

func (s *Store) UpdateWorkerHeartbeat(id string, runningJobs int32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if w, ok := s.Workers[id]; ok {
		w.LastHeartbeat = time.Now()
		w.RunningJobs = runningJobs
		if runningJobs > 0 {
			w.Status = StatusBusy
		} else {
			w.Status = StatusIdle
		}
	}
}

// Simple reaper to mark nodes offline
func (s *Store) ReapDeadWorkers(timeout time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for _, w := range s.Workers {
		if now.Sub(w.LastHeartbeat) > timeout {
			w.Status = StatusOffline
		}
	}
}
