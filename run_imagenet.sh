echo "----run_imagenet----";
echo "loss_type: $1"
echo "lamda_ini: $2"
echo "wgt_bit: $3"
echo "act_bit: $4"
echo "act_bit: $5"

for i in  0 1 2 4 8 32 64 128
do
   echo "spr_weight: $i"
   python main.py --config $5 optim.spr_w $i optim.loss_type $1  optim.lamda_ini $2 model.wgt_bit  $3 model.act_bit  $4
done

echo "----done---";
