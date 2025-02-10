# Transformer Model Implementation from Scratch

A PyTorch implementation of the Transformer architecture described in "Attention Is All You Need" (Vaswani et al., 2017), featuring complete component validation and modular design.

![Transformer Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*BHzGVskWGS_3jEcYYi6miQ.png)



## Features
- Full Transformer architecture with encoder-decoder structure
- Modular components with individual validation tests
- Positional encoding visualization
- Layer normalization with learnable parameters
- Multi-head attention mechanism
- Residual connections with dropout
- Xavier parameter initialization

## Implementation Details

### Core Components
#### **Input Embeddings**
\`InputEmbeddings\` converts token indices to scaled dense vectors.

#### **Positional Encoding**
\`PositionalEncoding\` adds positional information using sine/cosine functions.

#### **Layer Normalization**
Custom \`LayerNormalization\` implementation with epsilon stabilization.

#### **Feed Forward Block**
Two-layer network with ReLU activation and dropout.

#### **Residual Connection**
Skip connections with layer norm and dropout.

#### **Multi-Head Attention**
Parallel attention heads with scaled dot-product attention.

#### **Encoder/Decoder Blocks**
Stacked self-attention and feed-forward layers.

#### **Projection Layer**
Final linear transformation to vocabulary size.

## Validation Results

| Component           | Tests Passed  |
|---------------------|--------------|
| Input Embeddings   | 1/1 ✅       |
| Positional Encoding | 5/5 ✅       |
| Layer Normalization | 4/4 ✅       |
| Multi-Head Attention | Built-in validation |

### Positional Encoding Heatmap

(Insert visualization if needed)
## Contributing

1. Fork the repository.
2. Create a feature branch:
    \`\`\`bash
    git checkout -b feature-branch
    \`\`\`
3. Commit changes:
    \`\`\`bash
    git commit -am 'Add feature'
    \`\`\`
4. Push to branch:
    \`\`\`bash
    git push origin feature-branch
    \`\`\`
5. Open a Pull Request.

## License

This project is licensed under the MIT License.

## References

- Vaswani et al. *Attention Is All You Need*
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- Original Transformer implementation details

