package embedder

import (
	"bytes"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"
)

func msg(fields ...[]byte) []byte { return bytes.Join(fields, nil) }

func bytesField(num protowire.Number, v []byte) []byte {
	return protowire.AppendBytes(protowire.AppendTag(nil, num, protowire.BytesType), v)
}

func varintField(num protowire.Number, v uint64) []byte {
	return protowire.AppendVarint(protowire.AppendTag(nil, num, protowire.VarintType), v)
}

func valueInfo(name string, dims ...[]byte) []byte {
	shape := msg()
	for _, d := range dims {
		shape = append(shape, bytesField(onnxShapeDim, d)...)
	}
	tensor := msg(varintField(1, 7), bytesField(onnxTensorShape, shape))
	typ := bytesField(onnxTypeTensorType, tensor)
	return msg(bytesField(1, []byte(name)), bytesField(onnxValueInfoType, typ))
}

func TestFixModelDims(t *testing.T) {
	param := func(p string) []byte { return bytesField(onnxDimParam, []byte(p)) }
	value := func(v uint64) []byte { return varintField(onnxDimValue, v) }
	initializer := bytesField(5, bytes.Repeat([]byte{0xAB}, 64))
	graph := msg(
		initializer,
		bytesField(onnxGraphInput, valueInfo("input_ids", param("batch_size"), param("sequence_length"))),
		bytesField(onnxGraphOutput, valueInfo("sentence_embedding", param("batch_size"), param("other"))),
	)
	model := msg(varintField(1, 8), bytesField(onnxModelGraph, graph))

	out, n, err := fixModelDims(model, fixedShapeDims())
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Fatalf("fixed %d dims, want 3", n)
	}
	wantGraph := msg(
		initializer,
		bytesField(onnxGraphInput, valueInfo("input_ids", value(1), value(maxSeqLen))),
		bytesField(onnxGraphOutput, valueInfo("sentence_embedding", value(1), param("other"))),
	)
	want := msg(varintField(1, 8), bytesField(onnxModelGraph, wantGraph))
	if !bytes.Equal(out, want) {
		t.Fatalf("rewritten model differs:\n got %x\nwant %x", out, want)
	}
}

func TestFixModelDimsRejectsGarbage(t *testing.T) {
	if _, _, err := fixModelDims([]byte{0x3a, 0xff}, fixedShapeDims()); err == nil {
		t.Fatal("want a parse error for a truncated field")
	}
}
