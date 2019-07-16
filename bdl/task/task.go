package task

import (
	pb "github.com/j-haj/bdl/task_service"
)

type TaskID string

type Task struct {
	id TaskID
	task *pb.Task
}
