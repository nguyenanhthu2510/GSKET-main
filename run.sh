source activate thuna  

epochs=(100 500 1000)

# Loop through each epoch
for e in "${epochs[@]}"
do
    python main_knowledge.py --cfg config/iu_retrieval.yml --gpu 0 --version 1 --epoch "$e"
done
