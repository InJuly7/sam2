import onnx

def show(path):
    m = onnx.load(path)
    print(f"\n== {path} ==")
    for inp in m.graph.input:
        dims = inp.type.tensor_type.shape.dim
        def one(d):
            if d.dim_param:
                return f"param:{d.dim_param}"
            if d.dim_value:
                return f"value:{d.dim_value}"
            return "?"
        print(inp.name, [one(d) for d in dims])

show("/data/model/sam2/image_decoder_s.onnx")
show("/data/model/sam2/image_decoder_s_slim.onnx")