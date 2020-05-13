# !/bin/bash
split=data/msimut_small
[ -d "$split" ] || mkdir "$split"

cd data/msimut/
n=0
cnt=0
echo $(pwd)
for i in * 
do
    echo $i
    if [ $((n+=1)) -gt 1000 ]; then
        n=1
    fi

    if [ $((cnt+=1)) -gt 5000 ]; then
        break
    fi

    todir=../msimut_small/
    cp "$i" "$todir" 
done