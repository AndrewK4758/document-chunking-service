package server

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	pb "github.com/AndrewK4758/shared_protos"
)

func TestEmpirical_NatsWorker_EdgeCases(t *testing.T) {
	server := &NatsServer{}

	t.Run("EmptyIdentity_Panics", func(t *testing.T) {
		req := &pb.ExecuteTaskRequest{
			StepId:      "step-1",
			Identity:    &pb.InfrastructureIdentity{},
			TaskSubject: "worker.doc.inspect",
			Payload:     &structpb.Struct{},
		}
		b, _ := proto.Marshal(req)
		msg := &mockMsgForTerm{data: b, subject: "worker.doc.test"}
		assert.Panics(t, func() {
			server.processMessage(context.Background(), msg)
		})
	})

	t.Run("PartialIdentity_MissingJobId_Panics", func(t *testing.T) {
		req := &pb.ExecuteTaskRequest{
			StepId: "step-1",
			Identity: &pb.InfrastructureIdentity{
				TenantId: "t1",
				AppId:    "a1",
				JobId:    "",
			},
			TaskSubject: "worker.doc.inspect",
			Payload:     &structpb.Struct{},
		}
		b, _ := proto.Marshal(req)
		msg := &mockMsgForTerm{data: b, subject: "worker.doc.test"}
		assert.Panics(t, func() {
			server.processMessage(context.Background(), msg)
		})
	})

	t.Run("PayloadWithNonStringDocumentUri_Panics", func(t *testing.T) {
		payload, err := structpb.NewStruct(map[string]interface{}{
			"document_uri": 12345, // integer instead of string
		})
		assert.NoError(t, err)

		req := &pb.ExecuteTaskRequest{
			StepId: "step-1",
			Identity: &pb.InfrastructureIdentity{
				TenantId: "t1",
				AppId:    "a1",
				JobId:    "j1",
			},
			TaskSubject: "worker.doc.inspect",
			Payload:     payload,
		}
		b, _ := proto.Marshal(req)
		msg := &mockMsgForTerm{data: b, subject: "worker.doc.test"}
		assert.Panics(t, func() {
			server.processMessage(context.Background(), msg)
		})
	})

	t.Run("PayloadWithBooleanDocumentUri_Panics", func(t *testing.T) {
		payload, err := structpb.NewStruct(map[string]interface{}{
			"document_uri": true,
		})
		assert.NoError(t, err)

		req := &pb.ExecuteTaskRequest{
			StepId: "step-1",
			Identity: &pb.InfrastructureIdentity{
				TenantId: "t1",
				AppId:    "a1",
				JobId:    "j1",
			},
			TaskSubject: "worker.doc.inspect",
			Payload:     payload,
		}
		b, _ := proto.Marshal(req)
		msg := &mockMsgForTerm{data: b, subject: "worker.doc.test"}
		assert.Panics(t, func() {
			server.processMessage(context.Background(), msg)
		})
	})
}
