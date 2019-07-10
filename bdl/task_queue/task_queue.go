package task_queue

import (
	pbTask "github.com/j-haj/bdl/task"
	"sync"
)

type taskNode struct {
	next *taskNode
	data pbTask.Task
}

// FIFO queue implemented with a singly linked list
type TaskQueue struct {
	head *taskNode
	tail *taskNode
	size UInt
	mu   sync.Mutex
}

// Push adds a task to the back of the queue.
func (q *TaskQueue) Push(task *pbTask.Task) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if head == nil && tail == nil {
		head = &taskNode{nil, task}
		tail = head
	} else {
		tail.next = &taskNode{nil, task}
	}
	size++
}

// Pop returns the task at the front of the queue or an error if the queue is empty.
func (q *TaskQueue) Pop() (*pbTask.Task, error) {
	q.mu.Lock()
	q.mu.Unlock()
	if q.size == 0 {
		return nil, fmt.Errorf("cannot pop from an empty queue")
	}
	task := q.head
	task.next = nil
	q.head = q.head.next
	return task.data, nil
}

func (q *TaskQueue) isEmpty() bool {
	return q.size == 0
}

func (q *TaskQueue) Size() UInt {
	return q.size
}
