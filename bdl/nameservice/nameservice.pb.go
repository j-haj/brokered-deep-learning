// Code generated by protoc-gen-go. DO NOT EDIT.
// source: nameservice/nameservice.proto

package nameservice // import "github.com/j-haj/bdl/nameservice"

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"

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

type RegistrationRequest struct {
	Address string `protobuf:"bytes,1,opt,name=address" json:"address,omitempty"`
	// Location specifies a physical location like an AWS data center or region
	Location             string   `protobuf:"bytes,2,opt,name=location" json:"location,omitempty"`
	Types                []string `protobuf:"bytes,3,rep,name=types" json:"types,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *RegistrationRequest) Reset()         { *m = RegistrationRequest{} }
func (m *RegistrationRequest) String() string { return proto.CompactTextString(m) }
func (*RegistrationRequest) ProtoMessage()    {}
func (*RegistrationRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_nameservice_0c99f2c8e8236c9b, []int{0}
}
func (m *RegistrationRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RegistrationRequest.Unmarshal(m, b)
}
func (m *RegistrationRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RegistrationRequest.Marshal(b, m, deterministic)
}
func (dst *RegistrationRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegistrationRequest.Merge(dst, src)
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

func (m *RegistrationRequest) GetTypes() []string {
	if m != nil {
		return m.Types
	}
	return nil
}

type RegistrationResponse struct {
	// ID assigned to the registered broker.
	Id                   int64    `protobuf:"varint,1,opt,name=id" json:"id,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *RegistrationResponse) Reset()         { *m = RegistrationResponse{} }
func (m *RegistrationResponse) String() string { return proto.CompactTextString(m) }
func (*RegistrationResponse) ProtoMessage()    {}
func (*RegistrationResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_nameservice_0c99f2c8e8236c9b, []int{1}
}
func (m *RegistrationResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RegistrationResponse.Unmarshal(m, b)
}
func (m *RegistrationResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RegistrationResponse.Marshal(b, m, deterministic)
}
func (dst *RegistrationResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegistrationResponse.Merge(dst, src)
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
	Location             string   `protobuf:"bytes,1,opt,name=location" json:"location,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *BrokerRequest) Reset()         { *m = BrokerRequest{} }
func (m *BrokerRequest) String() string { return proto.CompactTextString(m) }
func (*BrokerRequest) ProtoMessage()    {}
func (*BrokerRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_nameservice_0c99f2c8e8236c9b, []int{2}
}
func (m *BrokerRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_BrokerRequest.Unmarshal(m, b)
}
func (m *BrokerRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_BrokerRequest.Marshal(b, m, deterministic)
}
func (dst *BrokerRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_BrokerRequest.Merge(dst, src)
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
	Address              string   `protobuf:"bytes,1,opt,name=address" json:"address,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *BrokerInfo) Reset()         { *m = BrokerInfo{} }
func (m *BrokerInfo) String() string { return proto.CompactTextString(m) }
func (*BrokerInfo) ProtoMessage()    {}
func (*BrokerInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_nameservice_0c99f2c8e8236c9b, []int{3}
}
func (m *BrokerInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_BrokerInfo.Unmarshal(m, b)
}
func (m *BrokerInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_BrokerInfo.Marshal(b, m, deterministic)
}
func (dst *BrokerInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_BrokerInfo.Merge(dst, src)
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

func init() {
	proto.RegisterType((*RegistrationRequest)(nil), "nameservice.RegistrationRequest")
	proto.RegisterType((*RegistrationResponse)(nil), "nameservice.RegistrationResponse")
	proto.RegisterType((*BrokerRequest)(nil), "nameservice.BrokerRequest")
	proto.RegisterType((*BrokerInfo)(nil), "nameservice.BrokerInfo")
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
	// RequestBroker returns a random broker in the request location.
	RequestBroker(context.Context, *BrokerRequest) (*BrokerInfo, error)
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
			MethodName: "RequestBroker",
			Handler:    _BrokerNameService_RequestBroker_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "nameservice/nameservice.proto",
}

func init() {
	proto.RegisterFile("nameservice/nameservice.proto", fileDescriptor_nameservice_0c99f2c8e8236c9b)
}

var fileDescriptor_nameservice_0c99f2c8e8236c9b = []byte{
	// 270 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x91, 0x51, 0x4b, 0xc3, 0x30,
	0x10, 0xc7, 0xd7, 0x15, 0x75, 0x3b, 0x99, 0xe0, 0x39, 0xb0, 0x14, 0x84, 0x9a, 0x87, 0x31, 0x10,
	0x5b, 0xd0, 0x6f, 0xb0, 0x07, 0xc1, 0x17, 0xc1, 0xf8, 0xe6, 0x5b, 0xda, 0x9c, 0x5b, 0xe6, 0xda,
	0xd4, 0x24, 0x13, 0xfc, 0x52, 0x7e, 0x46, 0xb1, 0x71, 0x92, 0xc2, 0xf4, 0x2d, 0x97, 0xfb, 0x25,
	0xff, 0xff, 0xff, 0x0e, 0x2e, 0x1a, 0x51, 0x93, 0x25, 0xf3, 0xae, 0x2a, 0x2a, 0x82, 0x73, 0xde,
	0x1a, 0xed, 0x34, 0x1e, 0x07, 0x57, 0x4c, 0xc0, 0x19, 0xa7, 0xa5, 0xb2, 0xce, 0x08, 0xa7, 0x74,
	0xc3, 0xe9, 0x6d, 0x4b, 0xd6, 0x61, 0x02, 0x47, 0x42, 0x4a, 0x43, 0xd6, 0x26, 0x51, 0x16, 0xcd,
	0xc7, 0x7c, 0x57, 0x62, 0x0a, 0xa3, 0x8d, 0xae, 0x3a, 0x38, 0x19, 0x76, 0xad, 0xdf, 0x1a, 0xa7,
	0x70, 0xe0, 0x3e, 0x5a, 0xb2, 0x49, 0x9c, 0xc5, 0xf3, 0x31, 0xf7, 0x05, 0x9b, 0xc1, 0xb4, 0x2f,
	0x61, 0x5b, 0xdd, 0x58, 0xc2, 0x13, 0x18, 0x2a, 0xd9, 0x7d, 0x1f, 0xf3, 0xa1, 0x92, 0xec, 0x0a,
	0x26, 0x0b, 0xa3, 0x5f, 0xc9, 0xec, 0x4c, 0x84, 0x52, 0x51, 0x5f, 0x8a, 0xcd, 0x00, 0x3c, 0x7c,
	0xdf, 0xbc, 0xe8, 0xbf, 0xed, 0xde, 0x7c, 0x46, 0x70, 0xea, 0xc1, 0x07, 0x51, 0xd3, 0x93, 0x4f,
	0x8d, 0x8f, 0x30, 0xf2, 0x96, 0xc8, 0x60, 0x96, 0x87, 0x23, 0xda, 0x33, 0x8c, 0xf4, 0xf2, 0x1f,
	0xc2, 0x67, 0x61, 0x03, 0xbc, 0x83, 0xc9, 0x0f, 0xef, 0xe5, 0x30, 0xed, 0xbd, 0xea, 0x25, 0x4b,
	0xcf, 0xf7, 0xf4, 0xbe, 0x83, 0xb0, 0xc1, 0x82, 0x3d, 0x67, 0x4b, 0xe5, 0x56, 0xdb, 0x32, 0xaf,
	0x74, 0x5d, 0xac, 0xaf, 0x57, 0x62, 0x5d, 0x94, 0x72, 0x13, 0xee, 0xb1, 0x3c, 0xec, 0x16, 0x79,
	0xfb, 0x15, 0x00, 0x00, 0xff, 0xff, 0x32, 0x00, 0xa3, 0x09, 0xe9, 0x01, 0x00, 0x00,
}
