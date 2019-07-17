package task_queue;

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	pb "github.com/j-haj/bdl/task"
)

var emptyTask = &pb.Task{TaskId: "localhost:10000#0", Source: "test"}

func buildTaskId(i int) string {
	return fmt.Sprintf("localhost:10000#%d", i)
}

func TestInitializedQueueIsEmpty(t *testing.T) {
	q := &TaskQueue{}
	assert.Equal(t, q.size, 0, "they should be equal")

}

func TestPush(t *testing.T) {
	q := &TaskQueue{}
	q.Push(emptyTask)

	assert.Equal(t, q.size, 1, "they should be equal")
	assert.NotNil(t, q.head)
	assert.NotNil(t, q.tail)
}

func TestPushFront(t *testing.T) {
	q := &TaskQueue{}
	q.PushFront(emptyTask)

	assert.Equal(t, q.size, 1, "they should be equal")
	assert.NotNil(t, q.head)
	assert.NotNil(t, q.tail)

	t1 := &pb.Task{TaskId: buildTaskId(3), Source: "test"}
	q.PushFront(t1)
	t2, err := q.Pop()
	assert.Nil(t, err)
	assert.Equal(t, t1, t2, "they should be equal")
	t3, err := q.Pop()
	assert.Nil(t, err)
	assert.Equal(t, t3, emptyTask, "they should be equal")
}

func TestPop(t *testing.T) {
	q := &TaskQueue{}
	q.Push(emptyTask)

	assert.Equal(t, q.size, 1, "they should be equal")

	task, err := q.Pop()

	assert.Nil(t, err)
	assert.Equal(t, task, emptyTask, "they should be equal")
}	

func TestSize(t *testing.T) {
	q := &TaskQueue{}
	for i := 0; i < 10; i++ {
		q.Push(&pb.Task{TaskId: buildTaskId(i), Source: "test"})
		assert.Equal(t, q.Size(), i+1, "they should be equal")
	}

	for i := 0; i < 10; i++ {
		v, err := q.Pop()
		assert.Equal(t, q.Size(), 9-i, "they should be equal")
		assert.Nil(t, err)
		assert.Equal(t, v.GetTaskId(), buildTaskId(i), "they should be equal")
	}
}

func TestEmpty(t *testing.T) {
	q := &TaskQueue{}
	assert.Equal(t, q.Empty(), true, "they should be equal")
}

func TestRemove(t *testing.T) {
	q := &TaskQueue{}
	for i := 0; i < 10; i++ {
		q.Push(&pb.Task{TaskId: buildTaskId(i), Source: "test"})
	}

	q.Remove(TaskID(buildTaskId(3)))
	assert.Equal(t, q.Size(), 9, "they should be equal")

	// There is no 3 remaining in the queue, this should be idempotent
	q.Remove(TaskID(buildTaskId(3)))
	assert.Equal(t, q.Size(), 9, "they should be equal")
	
	for i := 0; i < 9; i++ {
		v, err := q.Pop()
		assert.NotEqual(t, v.GetTaskId(), buildTaskId(3), "they should not be equal")
		assert.Nil(t, err)
	}


}
