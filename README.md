LLM powered lossless text compressor
===========================

Modified from [NNCP](https://bellard.org/nncp/), this project is to use Large Langauge Model to compress text data losslessly, for beating general purpose compression software like GZip and XZ. 

The project is implemented in pure Python, leveraging the Huggingface Transformer API. We provide an example showcasing how GPT-2 can be used to compress a small text corpus from Wikipedia.


To compress the text, run the following command:
```
bash scripts/eval.sh
```

For the example text file `unittest.txt` (size: 4KB),the script should generate a compressed file `unittest.bin` (size: 542 Bytes) and a decompressed file `decmp_unittest.txt`.

For reference, using XZ with the command  `xz-9 unittest.txt` produces a file `unittest.txt.xz` (size: 1.9KB), which is 3.5X larger than GPT2 based version. 

## Notes

- For reproducibility, it requires float64 numerical presion. So it runs super slow.

- You can run it with only CPUs. Just remove all `.cuda()` in the code.
