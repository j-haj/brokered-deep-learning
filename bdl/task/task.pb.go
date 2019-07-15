// Code generated by protoc-gen-go. DO NOT EDIT.
// source: task/task.proto

package task // import "github.com/j-haj/bdl/task"

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"
import result "github.com/j-haj/bdl/result"

import (
	context "golang.org/x/net/context"
	grpc "google.golang.org/grpc"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion2 // please upgrade the proto package

type Task struct {
	TaskId               string   `protobuf:"bytes,1,opt,name=task_id,json=taskId" json:"task_id,omitempty"`
	Source               string   `protobuf:"bytes,2,opt,name=source" json:"source,omitempty"`
	TaskObj              []byte   `protobuf:"bytes,3,opt,name=task_obj,json=taskObj,proto3" json:"task_obj,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Task) Reset()         { *m = Task{} }
func (m *Task) String() string { return proto.CompactTextString(m) }
func (*Task) ProtoMessage()    {}
func (*Task) Descriptor() ([]byte, []int) {
	return fileDescriptor_task_33ad9a4dae65b92e, []int{0}
}
func (m *Task) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Task.Unmarshal(m, b)
}
func (m *Task) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Task.Marshal(b, m, deterministic)
}
func (dst *Task) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Task.Merge(dst, src)
}
func (m *Task) XXX_Size() int {
	return xxx_messageInfo_Task.Size(m)
}
func (m *Task) XXX_DiscardUnknown() {
	xxx_messageInfo_Task.DiscardUnknown(m)
}

var xxx_messageInfo_Task proto.InternalMessageInfo

func (m *Task) GetTaskId() string {
	if m != nil {
		return m.TaskId
	}
	return ""
}

func (m *Task) GetSource() string {
	if m != nil {
		return m.Source
	}
	return ""
}

func (m *Task) GetTaskObj() []byte {
	if m != nil {
		return m.TaskObj
	}
	return nil
}

type TaskRequest struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *TaskRequest) Reset()         { *m = TaskRequest{} }
func (m *TaskRequest) String() string { return proto.CompactTextString(m) }
func (*TaskRequest) ProtoMessage()    {}
func (*TaskRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_task_33ad9a4dae65b92e, []int{1}
}
func (m *TaskRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_TaskRequest.Unmarshal(m, b)
}
func (m *TaskRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_TaskRequest.Marshal(b, m, deterministic)
}
func (dst *TaskRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TaskRequest.Merge(dst, src)
}
func (m *TaskRequest) XXX_Size() int {
	return xxx_messageInfo_TaskRequest.Size(m)
}
func (m *TaskRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_TaskRequest.DiscardUnknown(m)
}

var xxx_messageInfo_TaskRequest proto.InternalMessageInfo

type ResultResponse struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ResultResponse) Reset()         { *m = ResultResponse{} }
func (m *ResultResponse) String() string { return proto.CompactTextString(m) }
func (*ResultResponse) ProtoMessage()    {}
func (*ResultResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_task_33ad9a4dae65b92e, []int{2}
}
func (m *ResultResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ResultResponse.Unmarshal(m, b)
}
func (m *ResultResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ResultResponse.Marshal(b, m, deterministic)
}
func (dst *ResultResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ResultResponse.Merge(dst, src)
}
func (m *ResultResponse) XXX_Size() int {
	return xxx_messageInfo_ResultResponse.Size(m)
}
func (m *ResultResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_ResultResponse.DiscardUnknown(m)
}

var xxx_messageInfo_ResultResponse proto.InternalMessageInfo

func init() {
	proto.RegisterType((*Task)(nil), "task.Task")
	proto.RegisterType((*TaskRequest)(nil), "task.TaskRequest")
	proto.RegisterType((*ResultResponse)(nil), "task.ResultResponse")
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion4

// TaskServiceClient is the client API for TaskService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type TaskServiceClient interface {
	RequestTask(ctx context.Context, in *TaskRequest, opts ...grpc.CallOption) (*Task, error)
	SendResult(ctx context.Context, in *result.Result, opts ...grpc.CallOption) (*ResultResponse, error)
}

type taskServiceClient struct {
	cc *grpc.ClientConn
}

func NewTaskServiceClient(cc *grpc.ClientConn) TaskServiceClient {
	return &taskServiceClient{cc}
}

func (c *taskServiceClient) RequestTask(ctx context.Context, in *TaskRequest, opts ...grpc.CallOption) (*Task, error) {
	out := new(Task)
	err := c.cc.Invoke(ctx, "/task.TaskService/RequestTask", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *taskServiceClient) SendResult(ctx context.Context, in *result.Result, opts ...grpc.CallOption) (*ResultResponse, error) {
	out := new(ResultResponse)
	err := c.cc.Invoke(ctx, "/task.TaskService/SendResult", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// TaskServiceServer is the server API for TaskService service.
type TaskServiceServer interface {
	RequestTask(context.Context, *TaskRequest) (*Task, error)
	SendResult(context.Context, *result.Result) (*ResultResponse, error)
}

func RegisterTaskServiceServer(s *grpc.Server, srv TaskServiceServer) {
	s.RegisterService(&_TaskService_serviceDesc, srv)
}

func _TaskService_RequestTask_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(TaskRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TaskServiceServer).RequestTask(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/task.TaskService/RequestTask",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TaskServiceServer).RequestTask(ctx, req.(*TaskRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _TaskService_SendResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(result.Result)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TaskServiceServer).SendResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/task.TaskService/SendResult",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TaskServiceServer).SendResult(ctx, req.(*result.Result))
	}
	return interceptor(ctx, in, info, handler)
}

var _TaskService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "task.TaskService",
	HandlerType: (*TaskServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "RequestTask",
			Handler:    _TaskService_RequestTask_Handler,
		},
		{
			MethodName: "SendResult",
			Handler:    _TaskService_SendResult_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "task/task.proto",
}

func init() { proto.RegisterFile("task/task.proto", fileDescriptor_task_33ad9a4dae65b92e) }

var fileDescriptor_task_33ad9a4dae65b92e = []byte{
	// 235 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x54, 0x90, 0x31, 0x4f, 0xc3, 0x30,
	0x10, 0x85, 0x1b, 0xa8, 0x02, 0x5c, 0xa1, 0x80, 0x41, 0x90, 0x86, 0xa5, 0xf2, 0xd4, 0x05, 0x47,
	0x02, 0x7e, 0x01, 0x1b, 0x13, 0x92, 0xcb, 0xc4, 0x82, 0xe2, 0xe4, 0x44, 0x93, 0x96, 0xba, 0xf8,
	0x6c, 0x7e, 0x3f, 0xb2, 0x2f, 0x12, 0x74, 0xb1, 0xfd, 0xce, 0xcf, 0xdf, 0x9d, 0x1f, 0x9c, 0xfb,
	0x9a, 0xd6, 0x55, 0x5c, 0xd4, 0xce, 0x59, 0x6f, 0xc5, 0x38, 0x9e, 0xcb, 0x2b, 0x87, 0x14, 0x36,
	0xbe, 0xe2, 0x8d, 0xaf, 0xa4, 0x86, 0xf1, 0x5b, 0x4d, 0x6b, 0x71, 0x0b, 0x47, 0xd1, 0xf4, 0xd1,
	0xb5, 0x45, 0x36, 0xcf, 0x16, 0x27, 0x3a, 0x8f, 0xf2, 0xa5, 0x15, 0x37, 0x90, 0x93, 0x0d, 0xae,
	0xc1, 0xe2, 0x80, 0xeb, 0xac, 0xc4, 0x0c, 0x8e, 0xd3, 0x03, 0x6b, 0xfa, 0xe2, 0x70, 0x9e, 0x2d,
	0x4e, 0x75, 0x02, 0xbc, 0x9a, 0x5e, 0x9e, 0xc1, 0x24, 0x32, 0x35, 0x7e, 0x07, 0x24, 0x2f, 0x2f,
	0x60, 0xaa, 0x53, 0x4b, 0x8d, 0xb4, 0xb3, 0x5b, 0xc2, 0x07, 0x62, 0xc3, 0x12, 0xdd, 0x4f, 0xd7,
	0xa0, 0x50, 0x30, 0x19, 0xbc, 0x69, 0x94, 0x4b, 0x95, 0x46, 0xff, 0x87, 0x28, 0xe1, 0xaf, 0x24,
	0x47, 0xe2, 0x09, 0x60, 0x89, 0xdb, 0x96, 0xa1, 0x62, 0xaa, 0x86, 0x0f, 0xb1, 0x2e, 0xaf, 0xd9,
	0xbb, 0xdf, 0x52, 0x8e, 0x9e, 0xef, 0xde, 0x67, 0x9f, 0x9d, 0x5f, 0x05, 0xa3, 0x1a, 0xfb, 0x55,
	0xf5, 0xf7, 0xab, 0xba, 0xaf, 0x4c, 0xbb, 0x49, 0x39, 0x99, 0x3c, 0xa5, 0xf1, 0xf8, 0x1b, 0x00,
	0x00, 0xff, 0xff, 0x74, 0x13, 0x1b, 0x4f, 0x3b, 0x01, 0x00, 0x00,
}
