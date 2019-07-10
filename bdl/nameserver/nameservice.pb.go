// Code generated by protoc-gen-go. DO NOT EDIT.
// source: nameserver/nameservice.proto

package nameservice

import (
	context "context"
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
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

type RegistrationRequest struct {
	Address string `protobuf:"bytes,1,opt,name=address,proto3" json:"address,omitempty"`
	// Location specifies a physical location like an AWS data center or region
	Location             string   `protobuf:"bytes,2,opt,name=location,proto3" json:"location,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *RegistrationRequest) Reset()         { *m = RegistrationRequest{} }
func (m *RegistrationRequest) String() string { return proto.CompactTextString(m) }
func (*RegistrationRequest) ProtoMessage()    {}
func (*RegistrationRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_db68d7a6ab0e12c5, []int{0}
}

func (m *RegistrationRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RegistrationRequest.Unmarshal(m, b)
}
func (m *RegistrationRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RegistrationRequest.Marshal(b, m, deterministic)
}
func (m *RegistrationRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegistrationRequest.Merge(m, src)
}
func (m *RegistrationRequest) XXX_Size() int {
	return xxx_messageInfo_RegistrationRequest.Size(m)
}
func (m *RegistrationRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_RegistrationRequest.DiscardUnknown(m)
}

var xxx_messageInfo_RegistrationRequest proto.InternalMessageInfo

func (m *RegistrationRequest) GetAddress() string {
	if m != nil {
		return m.Address
	}
	return ""
}

func (m *RegistrationRequest) GetLocation() string {
	if m != nil {
		return m.Location
	}
	return ""
}

type RegistrationResponse struct {
	// ID assigned to the registered broker.
	Id                   int64    `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *RegistrationResponse) Reset()         { *m = RegistrationResponse{} }
func (m *RegistrationResponse) String() string { return proto.CompactTextString(m) }
func (*RegistrationResponse) ProtoMessage()    {}
func (*RegistrationResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_db68d7a6ab0e12c5, []int{1}
}

func (m *RegistrationResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RegistrationResponse.Unmarshal(m, b)
}
func (m *RegistrationResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RegistrationResponse.Marshal(b, m, deterministic)
}
func (m *RegistrationResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegistrationResponse.Merge(m, src)
}
func (m *RegistrationResponse) XXX_Size() int {
	return xxx_messageInfo_RegistrationResponse.Size(m)
}
func (m *RegistrationResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_RegistrationResponse.DiscardUnknown(m)
}

var xxx_messageInfo_RegistrationResponse proto.InternalMessageInfo

func (m *RegistrationResponse) GetId() int64 {
	if m != nil {
		return m.Id
	}
	return 0
}

type BrokerRequest struct {
	// Requested location of the broker.
	Location             string   `protobuf:"bytes,1,opt,name=location,proto3" json:"location,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *BrokerRequest) Reset()         { *m = BrokerRequest{} }
func (m *BrokerRequest) String() string { return proto.CompactTextString(m) }
func (*BrokerRequest) ProtoMessage()    {}
func (*BrokerRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_db68d7a6ab0e12c5, []int{2}
}

func (m *BrokerRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_BrokerRequest.Unmarshal(m, b)
}
func (m *BrokerRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_BrokerRequest.Marshal(b, m, deterministic)
}
func (m *BrokerRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_BrokerRequest.Merge(m, src)
}
func (m *BrokerRequest) XXX_Size() int {
	return xxx_messageInfo_BrokerRequest.Size(m)
}
func (m *BrokerRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_BrokerRequest.DiscardUnknown(m)
}

var xxx_messageInfo_BrokerRequest proto.InternalMessageInfo

func (m *BrokerRequest) GetLocation() string {
	if m != nil {
		return m.Location
	}
	return ""
}

type BrokerInfo struct {
	// Address:port of an available broker.
	Address              string   `protobuf:"bytes,1,opt,name=address,proto3" json:"address,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *BrokerInfo) Reset()         { *m = BrokerInfo{} }
func (m *BrokerInfo) String() string { return proto.CompactTextString(m) }
func (*BrokerInfo) ProtoMessage()    {}
func (*BrokerInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_db68d7a6ab0e12c5, []int{3}
}

func (m *BrokerInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_BrokerInfo.Unmarshal(m, b)
}
func (m *BrokerInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_BrokerInfo.Marshal(b, m, deterministic)
}
func (m *BrokerInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_BrokerInfo.Merge(m, src)
}
func (m *BrokerInfo) XXX_Size() int {
	return xxx_messageInfo_BrokerInfo.Size(m)
}
func (m *BrokerInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_BrokerInfo.DiscardUnknown(m)
}

var xxx_messageInfo_BrokerInfo proto.InternalMessageInfo

func (m *BrokerInfo) GetAddress() string {
	if m != nil {
		return m.Address
	}
	return ""
}

type HeartbeatRequest struct {
	// Broker ID
	Id                   int64    `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *HeartbeatRequest) Reset()         { *m = HeartbeatRequest{} }
func (m *HeartbeatRequest) String() string { return proto.CompactTextString(m) }
func (*HeartbeatRequest) ProtoMessage()    {}
func (*HeartbeatRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_db68d7a6ab0e12c5, []int{4}
}

func (m *HeartbeatRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_HeartbeatRequest.Unmarshal(m, b)
}
func (m *HeartbeatRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_HeartbeatRequest.Marshal(b, m, deterministic)
}
func (m *HeartbeatRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_HeartbeatRequest.Merge(m, src)
}
func (m *HeartbeatRequest) XXX_Size() int {
	return xxx_messageInfo_HeartbeatRequest.Size(m)
}
func (m *HeartbeatRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_HeartbeatRequest.DiscardUnknown(m)
}

var xxx_messageInfo_HeartbeatRequest proto.InternalMessageInfo

func (m *HeartbeatRequest) GetId() int64 {
	if m != nil {
		return m.Id
	}
	return 0
}

type HeartbeatResponse struct {
	// Reregister = true means the broker should re-regsiter with the nameserver. The
	// nameserver likely dropped the connection due to timeout.
	Reregister           bool     `protobuf:"varint,1,opt,name=reregister,proto3" json:"reregister,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *HeartbeatResponse) Reset()         { *m = HeartbeatResponse{} }
func (m *HeartbeatResponse) String() string { return proto.CompactTextString(m) }
func (*HeartbeatResponse) ProtoMessage()    {}
func (*HeartbeatResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_db68d7a6ab0e12c5, []int{5}
}

func (m *HeartbeatResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_HeartbeatResponse.Unmarshal(m, b)
}
func (m *HeartbeatResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_HeartbeatResponse.Marshal(b, m, deterministic)
}
func (m *HeartbeatResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_HeartbeatResponse.Merge(m, src)
}
func (m *HeartbeatResponse) XXX_Size() int {
	return xxx_messageInfo_HeartbeatResponse.Size(m)
}
func (m *HeartbeatResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_HeartbeatResponse.DiscardUnknown(m)
}

var xxx_messageInfo_HeartbeatResponse proto.InternalMessageInfo

func (m *HeartbeatResponse) GetReregister() bool {
	if m != nil {
		return m.Reregister
	}
	return false
}

func init() {
	proto.RegisterType((*RegistrationRequest)(nil), "nameservice.RegistrationRequest")
	proto.RegisterType((*RegistrationResponse)(nil), "nameservice.RegistrationResponse")
	proto.RegisterType((*BrokerRequest)(nil), "nameservice.BrokerRequest")
	proto.RegisterType((*BrokerInfo)(nil), "nameservice.BrokerInfo")
	proto.RegisterType((*HeartbeatRequest)(nil), "nameservice.HeartbeatRequest")
	proto.RegisterType((*HeartbeatResponse)(nil), "nameservice.HeartbeatResponse")
}

func init() { proto.RegisterFile("nameserver/nameservice.proto", fileDescriptor_db68d7a6ab0e12c5) }

var fileDescriptor_db68d7a6ab0e12c5 = []byte{
	// 304 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x92, 0x5f, 0x4b, 0xf3, 0x30,
	0x14, 0xc6, 0xd7, 0xbd, 0xf0, 0xda, 0x1d, 0x99, 0xb8, 0x28, 0x58, 0x8a, 0x8e, 0x9a, 0x0b, 0x11,
	0xc4, 0x16, 0xdc, 0x37, 0xd8, 0x85, 0x28, 0x8a, 0x60, 0xbc, 0xf3, 0x2e, 0x6d, 0x8f, 0x5b, 0xe6,
	0xda, 0xcc, 0x24, 0xf3, 0xd3, 0x7b, 0x21, 0xa6, 0x7f, 0x48, 0xe7, 0xe6, 0x5d, 0x93, 0xf3, 0x9c,
	0xe7, 0xfc, 0xfa, 0xe4, 0xc0, 0x69, 0xc9, 0x0b, 0xd4, 0xa8, 0x3e, 0x51, 0x25, 0xcd, 0xa7, 0xc8,
	0x30, 0x5e, 0x29, 0x69, 0x24, 0xd9, 0x77, 0xae, 0xe8, 0x03, 0x1c, 0x31, 0x9c, 0x09, 0x6d, 0x14,
	0x37, 0x42, 0x96, 0x0c, 0x3f, 0xd6, 0xa8, 0x0d, 0x09, 0x60, 0x8f, 0xe7, 0xb9, 0x42, 0xad, 0x03,
	0x2f, 0xf2, 0x2e, 0x07, 0xac, 0x39, 0x92, 0x10, 0xfc, 0xa5, 0xcc, 0xac, 0x38, 0xe8, 0xdb, 0x52,
	0x7b, 0xa6, 0x17, 0x70, 0xdc, 0x35, 0xd3, 0x2b, 0x59, 0x6a, 0x24, 0x07, 0xd0, 0x17, 0xb9, 0x35,
	0xfa, 0xc7, 0xfa, 0x22, 0xa7, 0x57, 0x30, 0x9c, 0x2a, 0xf9, 0x8e, 0xaa, 0x19, 0xe7, 0x9a, 0x7a,
	0xbf, 0x4c, 0xa1, 0x12, 0xdf, 0x97, 0x6f, 0x72, 0x37, 0x18, 0xa5, 0x70, 0x78, 0x87, 0x5c, 0x99,
	0x14, 0xb9, 0x69, 0x7c, 0x37, 0x07, 0x4f, 0x60, 0xe4, 0x68, 0x6a, 0xba, 0x31, 0x80, 0x42, 0x65,
	0xb9, 0x51, 0x59, 0xb1, 0xcf, 0x9c, 0x9b, 0x9b, 0x2f, 0x0f, 0x46, 0x15, 0xc1, 0x13, 0x2f, 0xf0,
	0xa5, 0x0a, 0x8e, 0x3c, 0x83, 0xcf, 0x6a, 0x05, 0x89, 0x62, 0x37, 0xe5, 0x2d, 0x79, 0x86, 0xe7,
	0x7f, 0x28, 0x2a, 0x0c, 0xda, 0x23, 0x8f, 0x30, 0x68, 0xe9, 0xc8, 0x59, 0xa7, 0x63, 0xf3, 0xcf,
	0xc2, 0xf1, 0xae, 0x72, 0xeb, 0x76, 0x0b, 0xc3, 0x5a, 0x5c, 0xc1, 0x93, 0xb0, 0xd3, 0xd2, 0x79,
	0x80, 0xf0, 0x64, 0x4b, 0xed, 0x27, 0x6f, 0xda, 0x9b, 0xd2, 0xd7, 0x68, 0x26, 0xcc, 0x7c, 0x9d,
	0xc6, 0x99, 0x2c, 0x92, 0xc5, 0xf5, 0x9c, 0x2f, 0x92, 0x34, 0x5f, 0xba, 0x8b, 0x95, 0xfe, 0xb7,
	0x9b, 0x35, 0xf9, 0x0e, 0x00, 0x00, 0xff, 0xff, 0x7b, 0x39, 0x2f, 0x1d, 0x79, 0x02, 0x00, 0x00,
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion4

// BrokerNameServiceClient is the client API for BrokerNameService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type BrokerNameServiceClient interface {
	// Register registers a broker with the nameserver.
	Register(ctx context.Context, in *RegistrationRequest, opts ...grpc.CallOption) (*RegistrationResponse, error)
	// Heartbeat is a keep alive request sent from a client broker to the nameserver.
	Heartbeat(ctx context.Context, in *HeartbeatRequest, opts ...grpc.CallOption) (*HeartbeatResponse, error)
	// RequestBroker returns a random broker in the request location.
	RequestBroker(ctx context.Context, in *BrokerRequest, opts ...grpc.CallOption) (*BrokerInfo, error)
}

type brokerNameServiceClient struct {
	cc *grpc.ClientConn
}

func NewBrokerNameServiceClient(cc *grpc.ClientConn) BrokerNameServiceClient {
	return &brokerNameServiceClient{cc}
}

func (c *brokerNameServiceClient) Register(ctx context.Context, in *RegistrationRequest, opts ...grpc.CallOption) (*RegistrationResponse, error) {
	out := new(RegistrationResponse)
	err := c.cc.Invoke(ctx, "/nameservice.BrokerNameService/Register", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *brokerNameServiceClient) Heartbeat(ctx context.Context, in *HeartbeatRequest, opts ...grpc.CallOption) (*HeartbeatResponse, error) {
	out := new(HeartbeatResponse)
	err := c.cc.Invoke(ctx, "/nameservice.BrokerNameService/Heartbeat", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *brokerNameServiceClient) RequestBroker(ctx context.Context, in *BrokerRequest, opts ...grpc.CallOption) (*BrokerInfo, error) {
	out := new(BrokerInfo)
	err := c.cc.Invoke(ctx, "/nameservice.BrokerNameService/RequestBroker", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// BrokerNameServiceServer is the server API for BrokerNameService service.
type BrokerNameServiceServer interface {
	// Register registers a broker with the nameserver.
	Register(context.Context, *RegistrationRequest) (*RegistrationResponse, error)
	// Heartbeat is a keep alive request sent from a client broker to the nameserver.
	Heartbeat(context.Context, *HeartbeatRequest) (*HeartbeatResponse, error)
	// RequestBroker returns a random broker in the request location.
	RequestBroker(context.Context, *BrokerRequest) (*BrokerInfo, error)
}

// UnimplementedBrokerNameServiceServer can be embedded to have forward compatible implementations.
type UnimplementedBrokerNameServiceServer struct {
}

func (*UnimplementedBrokerNameServiceServer) Register(ctx context.Context, req *RegistrationRequest) (*RegistrationResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Register not implemented")
}
func (*UnimplementedBrokerNameServiceServer) Heartbeat(ctx context.Context, req *HeartbeatRequest) (*HeartbeatResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Heartbeat not implemented")
}
func (*UnimplementedBrokerNameServiceServer) RequestBroker(ctx context.Context, req *BrokerRequest) (*BrokerInfo, error) {
	return nil, status.Errorf(codes.Unimplemented, "method RequestBroker not implemented")
}

func RegisterBrokerNameServiceServer(s *grpc.Server, srv BrokerNameServiceServer) {
	s.RegisterService(&_BrokerNameService_serviceDesc, srv)
}

func _BrokerNameService_Register_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RegistrationRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(BrokerNameServiceServer).Register(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/nameservice.BrokerNameService/Register",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(BrokerNameServiceServer).Register(ctx, req.(*RegistrationRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _BrokerNameService_Heartbeat_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(HeartbeatRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(BrokerNameServiceServer).Heartbeat(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/nameservice.BrokerNameService/Heartbeat",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(BrokerNameServiceServer).Heartbeat(ctx, req.(*HeartbeatRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _BrokerNameService_RequestBroker_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(BrokerRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(BrokerNameServiceServer).RequestBroker(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/nameservice.BrokerNameService/RequestBroker",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(BrokerNameServiceServer).RequestBroker(ctx, req.(*BrokerRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _BrokerNameService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "nameservice.BrokerNameService",
	HandlerType: (*BrokerNameServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Register",
			Handler:    _BrokerNameService_Register_Handler,
		},
		{
			MethodName: "Heartbeat",
			Handler:    _BrokerNameService_Heartbeat_Handler,
		},
		{
			MethodName: "RequestBroker",
			Handler:    _BrokerNameService_RequestBroker_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "nameserver/nameservice.proto",
}
