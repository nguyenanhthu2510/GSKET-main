source activate thuna  

epochs=(1)

# Loop through each epoch
for e in "${epochs[@]}"
do
    python main_knowledge.py --cfg config/iu_retrieval.yml --gpu 0 --version 1 --epoch "$e" --visual_extractor "swin" --batch_size 32

done