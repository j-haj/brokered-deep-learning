package task

import (
	pb "github.com/j-haj/bdl/task_service"
)

type TaskID string

type Task struct {
	Id TaskID
	TaskProto *pb.Task
}
