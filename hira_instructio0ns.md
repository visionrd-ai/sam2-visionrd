/home/multi-gpu/ai_res/Hira/sam2-visionrd/process_Sam2_hira_final.py

above is file for one folder only

to run this, use command:
python process_Sam2_hira_final.py --input-folder /home/multi-gpu/ai_res/Hira/sam2-visionrd/DATA/exp

python process_folder_hira.py \
    --data-folder /home/multi-gpu/ai_res/Hira/sam2-visionrd/DATA \
    --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
    --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
    --fps 20 \
    --device cuda


python process_folder_hira.py --data-folder /path/to/data --script /home/multi-gpu/ai_res/Hira/sam2-visionrd/process_bidirectional_combined.py --device cuda

python process_folder_hira.py \
  --data-folder /home/multi-gpu/ai_res/Hira/sam2-visionrd/DATA \
  --script process_bidirectional_combined.py \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --fps 20 \
  --device cuda \
  --continue-on-error \
  --output-folder /path/to/custom/output 