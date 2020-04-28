# !/bin/bash
split=data/msimut_split
[ -d "$split" ] || mkdir "$split"

cd data/msimut/
n=0
echo $(pwd)
for i in * 
do
    echo $i
    if [ $((n+=1)) -gt 1000 ]; then
        n=1
    fi
    todir=../msimut_split/msimut$n
    [ -d "$todir" ] || mkdir "$todir"
    cp "$i" "$todir" 
done 