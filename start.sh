#!/bin/bash
name="ffmpeg_tools.py"
# 查找 start.sh 的进程 ID
PID=$(pgrep -f $name)

# 检查是否找到进程
if [ -z "$PID" ]; then
    echo "没有找到正在运行的 start.sh 进程"
    exit 0
fi

# 结束进程
echo "正在停止进程 $PID..."
kill $PID

# 可选：检查进程是否已停止
sleep 1
if ps -p $PID > /dev/null; then
    echo "普通终止失败，尝试强制终止..."
    kill -9 $PID
fi

nohup python $name > output.log 2>&1 &

echo "$name 已重新启动"
