// Code generated by protoc-gen-go. DO NOT EDIT.
// source: broker_comm/broker_comm.proto

package broker_comm

import (
	context "context"
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	result "github.com/j-haj/bdl/result"
	task_service "github.com/j-haj/bdl/task_service"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

type AvailabilityInfo struct {
	BrokerId             string   `protobuf:"bytes,1,opt,name=broker_id,json=brokerId,proto3" json:"broker_id,omitempty"`
	Available            bool     `protobuf:"varint,2,opt,name=available,proto3" json:"available,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *AvailabilityInfo) Reset()         { *m = AvailabilityInfo{} }
func (m *AvailabilityInfo) String() string { return proto.CompactTextString(m) }
func (*AvailabilityInfo) ProtoMessage()    {}
func (*AvailabilityInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_3b1e38cf03a0803c, []int{0}
}

func (m *AvailabilityInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_AvailabilityInfo.Unmarshal(m, b)
}
func (m *AvailabilityInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_AvailabilityInfo.Marshal(b, m, deterministic)
}
func (m *AvailabilityInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_AvailabilityInfo.Merge(m, src)
}
func (m *AvailabilityInfo) XXX_Size() int {
	return xxx_messageInfo_AvailabilityInfo.Size(m)
}
func (m *AvailabilityInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_AvailabilityInfo.DiscardUnknown(m)
}

var xxx_messageInfo_AvailabilityInfo proto.InternalMessageInfo

func (m *AvailabilityInfo) GetBrokerId() string {
	if m != nil {
		return m.BrokerId
	}
	return ""
}

func (m *AvailabilityInfo) GetAvailable() bool {
	if m != nil {
		return m.Available
	}
	return false
}

type AvailabilityResponse struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *AvailabilityResponse) Reset()         { *m = AvailabilityResponse{} }
func (m *AvailabilityResponse) String() string { return proto.CompactTextString(m) }
func (*AvailabilityResponse) ProtoMessage()    {}
func (*AvailabilityResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_3b1e38cf03a0803c, []int{1}
}

func (m *AvailabilityResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_AvailabilityResponse.Unmarshal(m, b)
}
func (m *AvailabilityResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_AvailabilityResponse.Marshal(b, m, deterministic)
}
func (m *AvailabilityResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_AvailabilityResponse.Merge(m, src)
}
func (m *AvailabilityResponse) XXX_Size() int {
	return xxx_messageInfo_AvailabilityResponse.Size(m)
}
func (m *AvailabilityResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_AvailabilityResponse.DiscardUnknown(m)
}

var xxx_messageInfo_AvailabilityResponse proto.InternalMessageInfo

type ConnectionRequest struct {
	BrokerId             string   `protobuf:"bytes,1,opt,name=broker_id,json=brokerId,proto3" json:"broker_id,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ConnectionRequest) Reset()         { *m = ConnectionRequest{} }
func (m *ConnectionRequest) String() string { return proto.CompactTextString(m) }
func (*ConnectionRequest) ProtoMessage()    {}
func (*ConnectionRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_3b1e38cf03a0803c, []int{2}
}

func (m *ConnectionRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ConnectionRequest.Unmarshal(m, b)
}
func (m *ConnectionRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ConnectionRequest.Marshal(b, m, deterministic)
}
func (m *ConnectionRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ConnectionRequest.Merge(m, src)
}
func (m *ConnectionRequest) XXX_Size() int {
	return xxx_messageInfo_ConnectionRequest.Size(m)
}
func (m *ConnectionRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_ConnectionRequest.DiscardUnknown(m)
}

var xxx_messageInfo_ConnectionRequest proto.InternalMessageInfo

func (m *ConnectionRequest) GetBrokerId() string {
	if m != nil {
		return m.BrokerId
	}
	return ""
}

type ConnectionResponse struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ConnectionResponse) Reset()         { *m = ConnectionResponse{} }
func (m *ConnectionResponse) String() string { return proto.CompactTextString(m) }
func (*ConnectionResponse) ProtoMessage()    {}
func (*ConnectionResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_3b1e38cf03a0803c, []int{3}
}

func (m *ConnectionResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ConnectionResponse.Unmarshal(m, b)
}
func (m *ConnectionResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ConnectionResponse.Marshal(b, m, deterministic)
}
func (m *ConnectionResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ConnectionResponse.Merge(m, src)
}
func (m *ConnectionResponse) XXX_Size() int {
	return xxx_messageInfo_ConnectionResponse.Size(m)
}
func (m *ConnectionResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_ConnectionResponse.DiscardUnknown(m)
}

var xxx_messageInfo_ConnectionResponse proto.InternalMessageInfo

type DisconnectRequest struct {
	BrokerId             string   `protobuf:"bytes,1,opt,name=broker_id,json=brokerId,proto3" json:"broker_id,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *DisconnectRequest) Reset()         { *m = DisconnectRequest{} }
func (m *DisconnectRequest) String() string { return proto.CompactTextString(m) }
func (*DisconnectRequest) ProtoMessage()    {}
func (*DisconnectRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_3b1e38cf03a0803c, []int{4}
}

func (m *DisconnectRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_DisconnectRequest.Unmarshal(m, b)
}
func (m *DisconnectRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_DisconnectRequest.Marshal(b, m, deterministic)
}
func (m *DisconnectRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DisconnectRequest.Merge(m, src)
}
func (m *DisconnectRequest) XXX_Size() int {
	return xxx_messageInfo_DisconnectRequest.Size(m)
}
func (m *DisconnectRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_DisconnectRequest.DiscardUnknown(m)
}

var xxx_messageInfo_DisconnectRequest proto.InternalMessageInfo

func (m *DisconnectRequest) GetBrokerId() string {
	if m != nil {
		return m.BrokerId
	}
	return ""
}

type DisconnectResponse struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *DisconnectResponse) Reset()         { *m = DisconnectResponse{} }
func (m *DisconnectResponse) String() string { return proto.CompactTextString(m) }
func (*DisconnectResponse) ProtoMessage()    {}
func (*DisconnectResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_3b1e38cf03a0803c, []int{5}
}

func (m *DisconnectResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_DisconnectResponse.Unmarshal(m, b)
}
func (m *DisconnectResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_DisconnectResponse.Marshal(b, m, deterministic)
}
func (m *DisconnectResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DisconnectResponse.Merge(m, src)
}
func (m *DisconnectResponse) XXX_Size() int {
	return xxx_messageInfo_DisconnectResponse.Size(m)
}
func (m *DisconnectResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_DisconnectResponse.DiscardUnknown(m)
}

var xxx_messageInfo_DisconnectResponse proto.InternalMessageInfo

type ProcessResponse struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ProcessResponse) Reset()         { *m = ProcessResponse{} }
func (m *ProcessResponse) String() string { return proto.CompactTextString(m) }
func (*ProcessResponse) ProtoMessage()    {}
func (*ProcessResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_3b1e38cf03a0803c, []int{6}
}

func (m *ProcessResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ProcessResponse.Unmarshal(m, b)
}
func (m *ProcessResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ProcessResponse.Marshal(b, m, deterministic)
}
func (m *ProcessResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ProcessResponse.Merge(m, src)
}
func (m *ProcessResponse) XXX_Size() int {
	return xxx_messageInfo_ProcessResponse.Size(m)
}
func (m *ProcessResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_ProcessResponse.DiscardUnknown(m)
}

var xxx_messageInfo_ProcessResponse proto.InternalMessageInfo

type ShareRequest struct {
	OriginatorId         string             `protobuf:"bytes,1,opt,name=originator_id,json=originatorId,proto3" json:"originator_id,omitempty"`
	OriginatorAddress    string             `protobuf:"bytes,2,opt,name=originator_address,json=originatorAddress,proto3" json:"originator_address,omitempty"`
	TaskProto            *task_service.Task `protobuf:"bytes,3,opt,name=task_proto,json=taskProto,proto3" json:"task_proto,omitempty"`
	XXX_NoUnkeyedLiteral struct{}           `json:"-"`
	XXX_unrecognized     []byte             `json:"-"`
	XXX_sizecache        int32              `json:"-"`
}

func (m *ShareRequest) Reset()         { *m = ShareRequest{} }
func (m *ShareRequest) String() string { return proto.CompactTextString(m) }
func (*ShareRequest) ProtoMessage()    {}
func (*ShareRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_3b1e38cf03a0803c, []int{7}
}

func (m *ShareRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ShareRequest.Unmarshal(m, b)
}
func (m *ShareRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ShareRequest.Marshal(b, m, deterministic)
}
func (m *ShareRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ShareRequest.Merge(m, src)
}
func (m *ShareRequest) XXX_Size() int {
	return xxx_messageInfo_ShareRequest.Size(m)
}
func (m *ShareRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_ShareRequest.DiscardUnknown(m)
}

var xxx_messageInfo_ShareRequest proto.InternalMessageInfo

func (m *ShareRequest) GetOriginatorId() string {
	if m != nil {
		return m.OriginatorId
	}
	return ""
}

func (m *ShareRequest) GetOriginatorAddress() string {
	if m != nil {
		return m.OriginatorAddress
	}
	return ""
}

func (m *ShareRequest) GetTaskProto() *task_service.Task {
	if m != nil {
		return m.TaskProto
	}
	return nil
}

type ShareResponse struct {
	Ok                   bool     `protobuf:"varint,1,opt,name=ok,proto3" json:"ok,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ShareResponse) Reset()         { *m = ShareResponse{} }
func (m *ShareResponse) String() string { return proto.CompactTextString(m) }
func (*ShareResponse) ProtoMessage()    {}
func (*ShareResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_3b1e38cf03a0803c, []int{8}
}

func (m *ShareResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ShareResponse.Unmarshal(m, b)
}
func (m *ShareResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ShareResponse.Marshal(b, m, deterministic)
}
func (m *ShareResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ShareResponse.Merge(m, src)
}
func (m *ShareResponse) XXX_Size() int {
	return xxx_messageInfo_ShareResponse.Size(m)
}
func (m *ShareResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_ShareResponse.DiscardUnknown(m)
}

var xxx_messageInfo_ShareResponse proto.InternalMessageInfo

func (m *ShareResponse) GetOk() bool {
	if m != nil {
		return m.Ok
	}
	return false
}

func init() {
	proto.RegisterType((*AvailabilityInfo)(nil), "AvailabilityInfo")
	proto.RegisterType((*AvailabilityResponse)(nil), "AvailabilityResponse")
	proto.RegisterType((*ConnectionRequest)(nil), "ConnectionRequest")
	proto.RegisterType((*ConnectionResponse)(nil), "ConnectionResponse")
	proto.RegisterType((*DisconnectRequest)(nil), "DisconnectRequest")
	proto.RegisterType((*DisconnectResponse)(nil), "DisconnectResponse")
	proto.RegisterType((*ProcessResponse)(nil), "ProcessResponse")
	proto.RegisterType((*ShareRequest)(nil), "ShareRequest")
	proto.RegisterType((*ShareResponse)(nil), "ShareResponse")
}

func init() { proto.RegisterFile("broker_comm/broker_comm.proto", fileDescriptor_3b1e38cf03a0803c) }

var fileDescriptor_3b1e38cf03a0803c = []byte{
	// 409 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x8c, 0x92, 0xcf, 0x6e, 0xd4, 0x30,
	0x10, 0xc6, 0x93, 0x20, 0xc1, 0x66, 0x68, 0xb6, 0x9b, 0xd9, 0x82, 0x56, 0x01, 0xd4, 0x55, 0xb8,
	0xec, 0x01, 0xbc, 0xd0, 0x22, 0x71, 0x43, 0x2a, 0xe5, 0x92, 0x03, 0x52, 0x95, 0x72, 0xaf, 0xbc,
	0x89, 0x01, 0x2b, 0x89, 0x5d, 0x6c, 0x6f, 0x25, 0x1e, 0x84, 0x27, 0xe0, 0x45, 0xd1, 0xda, 0x81,
	0x78, 0xff, 0x08, 0xf5, 0x94, 0xf8, 0x1b, 0xcf, 0x97, 0x99, 0xef, 0x17, 0x78, 0xb1, 0x52, 0xb2,
	0x61, 0xea, 0xa6, 0x92, 0x5d, 0xb7, 0xf4, 0xde, 0xc9, 0xad, 0x92, 0x46, 0x66, 0x53, 0xc5, 0xf4,
	0xba, 0x35, 0x4b, 0xf7, 0xe8, 0xc5, 0x53, 0x43, 0x75, 0x73, 0xa3, 0x99, 0xba, 0xe3, 0x15, 0x5b,
	0xfa, 0x07, 0x77, 0x21, 0xff, 0x0c, 0x93, 0x8b, 0x3b, 0xca, 0x5b, 0xba, 0xe2, 0x2d, 0x37, 0x3f,
	0x0b, 0xf1, 0x55, 0xe2, 0x33, 0x88, 0x7b, 0x7b, 0x5e, 0xcf, 0xc2, 0x79, 0xb8, 0x88, 0xcb, 0x91,
	0x13, 0x8a, 0x1a, 0x9f, 0x43, 0x4c, 0x5d, 0x43, 0xcb, 0x66, 0xd1, 0x3c, 0x5c, 0x8c, 0xca, 0x41,
	0xc8, 0x9f, 0xc2, 0x89, 0x6f, 0x57, 0x32, 0x7d, 0x2b, 0x85, 0x66, 0xf9, 0x1b, 0x48, 0x2f, 0xa5,
	0x10, 0xac, 0x32, 0x5c, 0x8a, 0x92, 0xfd, 0x58, 0x33, 0x6d, 0xfe, 0xfb, 0x9d, 0xfc, 0x04, 0xd0,
	0xef, 0x18, 0x7c, 0x3e, 0x71, 0x5d, 0xb9, 0xc2, 0x7d, 0x7d, 0xfc, 0x8e, 0xde, 0x27, 0x85, 0xe3,
	0x2b, 0x25, 0x2b, 0xa6, 0xf5, 0x3f, 0xe9, 0x57, 0x08, 0x47, 0xd7, 0xdf, 0xa9, 0x62, 0x7f, 0x6d,
	0x5f, 0x42, 0x22, 0x15, 0xff, 0xc6, 0x05, 0x35, 0xd2, 0xb3, 0x3e, 0x1a, 0xc4, 0xa2, 0xc6, 0xd7,
	0x80, 0xde, 0x25, 0x5a, 0xd7, 0x8a, 0x69, 0x6d, 0x73, 0x89, 0xcb, 0x74, 0xa8, 0x5c, 0xb8, 0x02,
	0xbe, 0x05, 0xb0, 0x10, 0x6c, 0xf8, 0xb3, 0x07, 0xf3, 0x70, 0xf1, 0xf8, 0x0c, 0xc9, 0x16, 0x97,
	0x2f, 0x54, 0x37, 0x65, 0xbc, 0x91, 0xae, 0x2c, 0xa1, 0x53, 0x48, 0xfa, 0xb1, 0xdc, 0xa0, 0x38,
	0x86, 0x48, 0x36, 0x76, 0x98, 0x51, 0x19, 0xc9, 0xe6, 0xec, 0x77, 0x04, 0xc7, 0x85, 0x30, 0x4c,
	0x7d, 0xb4, 0x3b, 0x5f, 0xca, 0xae, 0xc3, 0x0f, 0x30, 0xb9, 0x66, 0xa2, 0xf6, 0x59, 0x60, 0x4a,
	0x76, 0x49, 0x67, 0x4f, 0xc8, 0x41, 0x5a, 0x01, 0xbe, 0x83, 0x47, 0x7d, 0xfa, 0x88, 0x64, 0x8f,
	0x5c, 0x36, 0x25, 0x07, 0xd8, 0x04, 0xf8, 0x1e, 0x60, 0xc8, 0x1a, 0x91, 0xec, 0xa1, 0xca, 0xa6,
	0xe4, 0x00, 0x8c, 0x00, 0x5f, 0x41, 0x6c, 0x77, 0xdc, 0xec, 0x8e, 0x09, 0xf1, 0x31, 0x64, 0x63,
	0xb2, 0xb5, 0x7e, 0x1e, 0xe0, 0x39, 0x24, 0x03, 0xbc, 0x75, 0x6b, 0x70, 0x4c, 0xfa, 0x9f, 0xde,
	0x9d, 0xb3, 0x09, 0xd9, 0x85, 0x1b, 0xac, 0x1e, 0xda, 0xc8, 0xcf, 0xff, 0x04, 0x00, 0x00, 0xff,
	0xff, 0x4c, 0xbd, 0x5d, 0x22, 0x46, 0x03, 0x00, 0x00,
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion4

// InterBrokerCommClient is the client API for InterBrokerComm service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type InterBrokerCommClient interface {
	// SendAvailability communicates a brokers availability for additional tasks.
	// It is used to be send and retract availability.
	SendAvailability(ctx context.Context, in *AvailabilityInfo, opts ...grpc.CallOption) (*AvailabilityResponse, error)
	// Connect establishes a link between two brokers, allowing them to share
	// work.
	Connect(ctx context.Context, in *ConnectionRequest, opts ...grpc.CallOption) (*ConnectionResponse, error)
	// Disconnect can be called by a broker to terminate a connection with a linked broker.
	Disconnect(ctx context.Context, in *DisconnectRequest, opts ...grpc.CallOption) (*DisconnectResponse, error)
	// ShareTask sends a task to a linked broker.
	ShareTask(ctx context.Context, in *ShareRequest, opts ...grpc.CallOption) (*ShareResponse, error)
	// ProcessResult either sends the result to the model or sends the result on to the
	// next broker.
	ProcessResult(ctx context.Context, in *result.Result, opts ...grpc.CallOption) (*ProcessResponse, error)
}

type interBrokerCommClient struct {
	cc *grpc.ClientConn
}

func NewInterBrokerCommClient(cc *grpc.ClientConn) InterBrokerCommClient {
	return &interBrokerCommClient{cc}
}

func (c *interBrokerCommClient) SendAvailability(ctx context.Context, in *AvailabilityInfo, opts ...grpc.CallOption) (*AvailabilityResponse, error) {
	out := new(AvailabilityResponse)
	err := c.cc.Invoke(ctx, "/InterBrokerComm/SendAvailability", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *interBrokerCommClient) Connect(ctx context.Context, in *ConnectionRequest, opts ...grpc.CallOption) (*ConnectionResponse, error) {
	out := new(ConnectionResponse)
	err := c.cc.Invoke(ctx, "/InterBrokerComm/Connect", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *interBrokerCommClient) Disconnect(ctx context.Context, in *DisconnectRequest, opts ...grpc.CallOption) (*DisconnectResponse, error) {
	out := new(DisconnectResponse)
	err := c.cc.Invoke(ctx, "/InterBrokerComm/Disconnect", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *interBrokerCommClient) ShareTask(ctx context.Context, in *ShareRequest, opts ...grpc.CallOption) (*ShareResponse, error) {
	out := new(ShareResponse)
	err := c.cc.Invoke(ctx, "/InterBrokerComm/ShareTask", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *interBrokerCommClient) ProcessResult(ctx context.Context, in *result.Result, opts ...grpc.CallOption) (*ProcessResponse, error) {
	out := new(ProcessResponse)
	err := c.cc.Invoke(ctx, "/InterBrokerComm/ProcessResult", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// InterBrokerCommServer is the server API for InterBrokerComm service.
type InterBrokerCommServer interface {
	// SendAvailability communicates a brokers availability for additional tasks.
	// It is used to be send and retract availability.
	SendAvailability(context.Context, *AvailabilityInfo) (*AvailabilityResponse, error)
	// Connect establishes a link between two brokers, allowing them to share
	// work.
	Connect(context.Context, *ConnectionRequest) (*ConnectionResponse, error)
	// Disconnect can be called by a broker to terminate a connection with a linked broker.
	Disconnect(context.Context, *DisconnectRequest) (*DisconnectResponse, error)
	// ShareTask sends a task to a linked broker.
	ShareTask(context.Context, *ShareRequest) (*ShareResponse, error)
	// ProcessResult either sends the result to the model or sends the result on to the
	// next broker.
	ProcessResult(context.Context, *result.Result) (*ProcessResponse, error)
}

// UnimplementedInterBrokerCommServer can be embedded to have forward compatible implementations.
type UnimplementedInterBrokerCommServer struct {
}

func (*UnimplementedInterBrokerCommServer) SendAvailability(ctx context.Context, req *AvailabilityInfo) (*AvailabilityResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendAvailability not implemented")
}
func (*UnimplementedInterBrokerCommServer) Connect(ctx context.Context, req *ConnectionRequest) (*ConnectionResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Connect not implemented")
}
func (*UnimplementedInterBrokerCommServer) Disconnect(ctx context.Context, req *DisconnectRequest) (*DisconnectResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Disconnect not implemented")
}
func (*UnimplementedInterBrokerCommServer) ShareTask(ctx context.Context, req *ShareRequest) (*ShareResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ShareTask not implemented")
}
func (*UnimplementedInterBrokerCommServer) ProcessResult(ctx context.Context, req *result.Result) (*ProcessResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ProcessResult not implemented")
}

func RegisterInterBrokerCommServer(s *grpc.Server, srv InterBrokerCommServer) {
	s.RegisterService(&_InterBrokerComm_serviceDesc, srv)
}

func _InterBrokerComm_SendAvailability_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(AvailabilityInfo)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(InterBrokerCommServer).SendAvailability(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/InterBrokerComm/SendAvailability",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(InterBrokerCommServer).SendAvailability(ctx, req.(*AvailabilityInfo))
	}
	return interceptor(ctx, in, info, handler)
}

func _InterBrokerComm_Connect_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ConnectionRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(InterBrokerCommServer).Connect(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/InterBrokerComm/Connect",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(InterBrokerCommServer).Connect(ctx, req.(*ConnectionRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _InterBrokerComm_Disconnect_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DisconnectRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(InterBrokerCommServer).Disconnect(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/InterBrokerComm/Disconnect",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(InterBrokerCommServer).Disconnect(ctx, req.(*DisconnectRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _InterBrokerComm_ShareTask_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ShareRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(InterBrokerCommServer).ShareTask(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/InterBrokerComm/ShareTask",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(InterBrokerCommServer).ShareTask(ctx, req.(*ShareRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _InterBrokerComm_ProcessResult_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(result.Result)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(InterBrokerCommServer).ProcessResult(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/InterBrokerComm/ProcessResult",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(InterBrokerCommServer).ProcessResult(ctx, req.(*result.Result))
	}
	return interceptor(ctx, in, info, handler)
}

var _InterBrokerComm_serviceDesc = grpc.ServiceDesc{
	ServiceName: "InterBrokerComm",
	HandlerType: (*InterBrokerCommServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "SendAvailability",
			Handler:    _InterBrokerComm_SendAvailability_Handler,
		},
		{
			MethodName: "Connect",
			Handler:    _InterBrokerComm_Connect_Handler,
		},
		{
			MethodName: "Disconnect",
			Handler:    _InterBrokerComm_Disconnect_Handler,
		},
		{
			MethodName: "ShareTask",
			Handler:    _InterBrokerComm_ShareTask_Handler,
		},
		{
			MethodName: "ProcessResult",
			Handler:    _InterBrokerComm_ProcessResult_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "broker_comm/broker_comm.proto",
}
