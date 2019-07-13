package task_queue

import (
	"fmt"
	"sync"
	
	pbTask "github.com/j-haj/bdl/task"
)

type TaskID int64

type taskNode struct {
	next *taskNode
	id TaskID
	data *pbTask.Task
}

// FIFO queue implemented with a singly linked list
type TaskQueue struct {
	head *taskNode
	tail *taskNode
	size int
	mu   sync.Mutex
}

// Push adds a task to the back of the queue.
func (q *TaskQueue) Push(task *pbTask.Task) {
	q.mu.Lock()
	defer q.mu.Unlock()
	t := &taskNode{nil, TaskID(task.GetTaskId()), task}
	if q.head == nil && q.tail == nil {
		q.head = t
		q.tail = q.head
	} else {
		q.tail.next = t
		q.tail = t;
	}
	q.size++
}

// Pop returns the task at the front of the queue or an error if the queue is empty.
func (q *TaskQueue) Pop() (*pbTask.Task, error) {
	q.mu.Lock()
	q.mu.Unlock()
	if q.size == 0 {
		return nil, fmt.Errorf("cannot pop from an empty queue")
	}
	task := q.head
	q.head = q.head.next
	q.size--
	return task.data, nil
}

// Remove removes the task from the queue. If the task is not in the queue the method exits
// without error.
func (q *TaskQueue) Remove(taskId TaskID) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.head == nil {
		return
	} else if q.head.id == taskId {
		if q.tail == q.head {
			q.tail = q.head.next
		}
		q.head = q.head.next
	} else {
		x := q.head
		y := x
		for x != nil && x.id != taskId {
			y = x
			x = x.next
		}
		if x == nil {
			return
		}

		y.next = x.next
	}
	q.size--	
}

func (q *TaskQueue) Empty() bool {
	return q.size == 0
}

func (q *TaskQueue) Size() int {
	return q.size
}
