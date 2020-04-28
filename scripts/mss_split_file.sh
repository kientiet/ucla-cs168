# !/bin/bash
split=data/mss_split
[ -d "$split" ] || mkdir "$split"

cd data/mss/
n=0
current=$(pwd)
echo $current
for i in * 
do
    echo $i
    if [ $((n+=1)) -gt 1000 ]; then
        n=1
    fi
    todir=../mss_split/mss$n
    [ -d "$todir" ] || mkdir "$todir"
    cp "$i" "$todir" 
done