package server

import (
	"context"
	"testing"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
)

type mockMsgForTerm struct {
	data    []byte
	subject string
	termed  bool
	acked   bool
}

func (m *mockMsgForTerm) Metadata() (*jetstream.MsgMetadata, error) {
	return nil, nil
}
func (m *mockMsgForTerm) Subject() string {
	if m.subject == "" {
		return "worker.doc.chunking"
	}
	return m.subject
}
func (m *mockMsgForTerm) Data() []byte {
	return m.data
}
func (m *mockMsgForTerm) Headers() nats.Header {
	return nil
}
func (m *mockMsgForTerm) Reply() string {
	return ""
}
func (m *mockMsgForTerm) Ack() error {
	m.acked = true
	return nil
}
func (m *mockMsgForTerm) DoubleAck(context.Context) error {
	return nil
}
func (m *mockMsgForTerm) Nak() error {
	return nil
}
func (m *mockMsgForTerm) NakWithDelay(delay time.Duration) error {
	return nil
}
func (m *mockMsgForTerm) InProgress() error {
	return nil
}
func (m *mockMsgForTerm) Term() error {
	m.termed = true
	return nil
}
func (m *mockMsgForTerm) TermWithReason(reason string) error {
	m.termed = true
	return nil
}

func TestProcessMessage_InvalidProtobufPayload_CallsTerm(t *testing.T) {
	server := &NatsServer{}

	// Invalid wire format protobuf payload
	invalidPayload := []byte{0xFF, 0xFF, 0xFF, 0xFF, 0x07, 0x00}
	msg := &mockMsgForTerm{
		data:    invalidPayload,
		subject: "worker.doc.test",
	}

	server.processMessage(context.Background(), msg)

	if !msg.termed {
		t.Fatalf("Expected msg.Term() to be called on invalid protobuf payload, but it was not")
	}
	if msg.acked {
		t.Fatalf("Expected msg.Ack() NOT to be called on invalid protobuf payload")
	}
}
