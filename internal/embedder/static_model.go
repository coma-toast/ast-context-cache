package embedder

import (
	"os"
	"path/filepath"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// fixedShapeModelFile is a copy of model.onnx whose batch and sequence dimensions are
// pinned to the embedder's fixed 1 x maxSeqLen input. With static shapes ONNX Runtime
// plans the graph once (about 2x faster on the CPU) and CoreML can compile all of it.
const fixedShapeModelFile = "model.static-1x128.onnx"

// ONNX protobuf field numbers on the path ModelProto.graph -> {input,output,value_info}
// -> type.tensor_type.shape.dim.
const (
	onnxModelGraph     protowire.Number = 7
	onnxGraphInput     protowire.Number = 11
	onnxGraphOutput    protowire.Number = 12
	onnxGraphValueInfo protowire.Number = 13
	onnxValueInfoType  protowire.Number = 2
	onnxTypeTensorType protowire.Number = 1
	onnxTensorShape    protowire.Number = 2
	onnxShapeDim       protowire.Number = 1
	onnxDimValue       protowire.Number = 1
	onnxDimParam       protowire.Number = 2
)

// fixedShapeDims maps the symbolic dimensions of the all-mpnet export to fixed sizes.
func fixedShapeDims() map[string]int64 {
	return map[string]int64{"batch_size": 1, "sequence_length": maxSeqLen}
}

// ensureFixedShapeModel returns the path of a fixed-shape copy of modelDir/model.onnx,
// writing it when missing or older than the source.
func ensureFixedShapeModel(modelDir string) (string, error) {
	src := filepath.Join(modelDir, "model.onnx")
	dst := filepath.Join(modelDir, fixedShapeModelFile)
	srcInfo, err := os.Stat(src)
	if err != nil {
		return "", errs.WrapMessage("failed to stat model.onnx", err, "path", src)
	}
	if dstInfo, err := os.Stat(dst); err == nil && !dstInfo.ModTime().Before(srcInfo.ModTime()) {
		return dst, nil
	}
	raw, err := os.ReadFile(src)
	if err != nil {
		return "", errs.WrapMessage("failed to read model.onnx", err, "path", src)
	}
	fixed, n, err := fixModelDims(raw, fixedShapeDims())
	if err != nil {
		return "", err
	}
	if n == 0 {
		return "", errs.New("model.onnx has no batch_size or sequence_length dimensions to fix")
	}
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, fixed, 0o644); err != nil {
		return "", errs.WrapMessage("failed to write fixed-shape model", err, "path", tmp)
	}
	if err := os.Rename(tmp, dst); err != nil {
		os.Remove(tmp)
		return "", errs.WrapMessage("failed to save fixed-shape model", err, "path", dst)
	}
	logger.Info("Wrote fixed-shape ONNX model", "path", dst, "dims_fixed", n)
	return dst, nil
}

// fixModelDims rewrites the named symbolic dimensions in a serialized ModelProto and
// reports how many it replaced. Every other field is copied byte for byte.
func fixModelDims(model []byte, dims map[string]int64) ([]byte, int, error) {
	count := 0
	dim := func(b []byte) ([]byte, error) {
		param, ok, err := stringField(b, onnxDimParam)
		if err != nil || !ok {
			return b, err
		}
		v, fix := dims[param]
		if !fix {
			return b, nil
		}
		count++
		return protowire.AppendVarint(protowire.AppendTag(nil, onnxDimValue, protowire.VarintType), uint64(v)), nil
	}
	shape := nested(map[protowire.Number]rewriter{onnxShapeDim: dim})
	tensor := nested(map[protowire.Number]rewriter{onnxTensorShape: shape})
	typ := nested(map[protowire.Number]rewriter{onnxTypeTensorType: tensor})
	valueInfo := nested(map[protowire.Number]rewriter{onnxValueInfoType: typ})
	graph := nested(map[protowire.Number]rewriter{
		onnxGraphInput: valueInfo, onnxGraphOutput: valueInfo, onnxGraphValueInfo: valueInfo,
	})
	out, err := nested(map[protowire.Number]rewriter{onnxModelGraph: graph})(model)
	if err != nil {
		return nil, 0, errs.WrapMessage("failed to rewrite model.onnx dimensions", err)
	}
	return out, count, nil
}

type rewriter func([]byte) ([]byte, error)

// nested returns a rewriter that applies fields[n] to each length-delimited field n of a
// message and copies all other fields unchanged.
func nested(fields map[protowire.Number]rewriter) rewriter {
	return func(b []byte) ([]byte, error) {
		out := make([]byte, 0, len(b))
		for len(b) > 0 {
			num, typ, tagLen := protowire.ConsumeTag(b)
			if tagLen < 0 {
				return nil, protowire.ParseError(tagLen)
			}
			valLen := protowire.ConsumeFieldValue(num, typ, b[tagLen:])
			if valLen < 0 {
				return nil, protowire.ParseError(valLen)
			}
			field := b[:tagLen+valLen]
			b = b[tagLen+valLen:]
			fn, ok := fields[num]
			if !ok || typ != protowire.BytesType {
				out = append(out, field...)
				continue
			}
			val, n := protowire.ConsumeBytes(field[tagLen:])
			if n < 0 {
				return nil, protowire.ParseError(n)
			}
			rewritten, err := fn(val)
			if err != nil {
				return nil, err
			}
			out = protowire.AppendTag(out, num, protowire.BytesType)
			out = protowire.AppendBytes(out, rewritten)
		}
		return out, nil
	}
}

// stringField returns the first string field num of a message.
func stringField(b []byte, num protowire.Number) (string, bool, error) {
	for len(b) > 0 {
		n, typ, tagLen := protowire.ConsumeTag(b)
		if tagLen < 0 {
			return "", false, protowire.ParseError(tagLen)
		}
		valLen := protowire.ConsumeFieldValue(n, typ, b[tagLen:])
		if valLen < 0 {
			return "", false, protowire.ParseError(valLen)
		}
		if n == num && typ == protowire.BytesType {
			v, _ := protowire.ConsumeBytes(b[tagLen:])
			return string(v), true, nil
		}
		b = b[tagLen+valLen:]
	}
	return "", false, nil
}
