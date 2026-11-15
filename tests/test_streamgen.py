import pytest
import torch
from streamgen import StreamGen, create_streamgen

class TestStreamGen:
    def test_init(self):
        m = StreamGen(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = StreamGen(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_streamgen()
        assert isinstance(m, StreamGen)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
