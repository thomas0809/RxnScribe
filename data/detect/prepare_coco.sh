cp -r /data/scratch/jiang_guo/chemie/diagram-parse/annotations/detect/images ./images

cd splits
ln -s ../images ./train2017
ln -s ../images ./val2017
