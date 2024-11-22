# CUDA_VISIBLE_DEVICES=3 python eval.py --output_file valid.txt.bin

input_file=unittest.txt
cmp_file=unittest.bin
decmp_file=decmp_unittest.txt

rm $cmp_file
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=0 python gpt2zip.py --input_file $input_file --output_file $cmp_file 
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=0 python gpt2zip.py --input_file $cmp_file --output_file $decmp_file --decompress




# cmp_file=single_input_test.bin
# rm $cmp_file
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=3 python unit_test.py --output_file $cmp_file 
# rm $cmp_file