import subprocess
import asyncio
import sys
import uuid
import os
import tempfile
import shutil
from pathlib import Path

from datetime import datetime
from logging.handlers import RotatingFileHandler

from typing import Generic, TypeVar, Optional, Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field

import aiohttp
import whisper
import math
from typing import Optional, List, NamedTuple
from moviepy import VideoFileClip, AudioFileClip

import oss2 as oss
from oss2.credentials import EnvironmentVariableCredentialsProvider
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import logging

logger = logging.getLogger(__name__)

app = FastAPI()

origins = ["*"]
# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 也可以使用 ["*"] 允许所有来源
    allow_credentials=True,  # 允许携带凭证（如cookies）
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
    # 还可以设置其他参数，如：
    # expose_headers: 指定浏览器可以访问的响应头
    # max_age: 设置浏览器缓存CORS响应的最长时间（秒）
)


# 配置日志记录器
def setup_logging():
    # 创建记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # 创建文件处理器（每天滚动，最多保留7天）
    file_handler = RotatingFileHandler(
        filename='app.log',
        maxBytes=1024 * 1024 * 5,  # 5MB
        backupCount=7
    )
    file_handler.setLevel(logging.INFO)
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # 将处理器添加到记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# 调用配置函数
setup_logging()


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return Result.from_exception(exc).to_response()


def create_directory(dir_path: Optional[str] = None) -> Path:
    """
    创建目录或临时目录

    参数:
        dir_path: 可选的目标目录路径 (None 表示创建临时目录)

    返回:
        创建的目录路径 (Path 对象)

    异常:
        FileExistsError: 当目标路径已存在但不是目录
        PermissionError: 当没有创建目录的权限
        OSError: 其他系统级错误
    """
    # 处理临时目录请求
    if dir_path is None:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        return Path(temp_dir)

    # 处理指定目录路径
    path = Path(dir_path).expanduser().resolve()

    # 如果目录已存在
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"路径已存在但不是目录: {path}")
        return path

    # 创建新目录
    try:
        path.mkdir(parents=True, exist_ok=False)
        return path
    except PermissionError:
        raise PermissionError(f"没有权限创建目录: {path}") from None
    except FileNotFoundError:
        raise FileNotFoundError(f"父目录不存在: {path.parent}") from None
    except OSError as e:
        raise OSError(f"创建目录失败: {e}") from e


def is_temp_directory(path: Path) -> bool:
    """
    判断路径是否在系统临时目录内（安全措施）

    参数:
        path: 要检查的路径

    返回:
        如果路径在系统临时目录内则返回 True
    """
    # 获取系统临时目录的规范路径
    system_temp_dir = Path(tempfile.gettempdir()).resolve()

    # 解析目标路径
    try:
        resolved_path = path.resolve()
    except OSError:
        # 路径解析失败（可能是无效路径）
        return False

    # 检查路径是否在临时目录内
    try:
        # 检查路径是否以临时目录开头
        return resolved_path.is_relative_to(system_temp_dir)
    except AttributeError:
        # Python < 3.9 兼容方案
        try:
            relative = resolved_path.relative_to(system_temp_dir)
            return True
        except ValueError:
            return False


def safe_delete_temp_directory(path: Path) -> bool:
    """
    安全删除目录（仅当在系统临时目录内时）

    参数:
        path: 要删除的目录路径

    返回:
        如果成功删除返回 True，否则返回 False
    """
    # 验证路径是否在临时目录内
    if not is_temp_directory(path):
        logger.info(f"安全警告: 不删除非临时目录 - {path}")
        return False

    # 验证路径是否存在且是目录
    if not path.exists():
        logger.info(f"目录不存在: {path}")
        return False

    if not path.is_dir():
        logger.info(f"路径不是目录: {path}")
        return False

    try:
        # 递归删除目录及其内容
        shutil.rmtree(path)
        logger.info(f"已安全删除临时目录: {path}")
        return True
    except Exception as e:
        logger.info(f"删除目录失败: {e}")
        return False


# 使用上下文管理器自动处理临时目录
class TempDirectory:
    """临时目录上下文管理器，自动创建和删除临时目录"""

    def __init__(self, prefix: str = "tmp_", suffix: str = ""):
        self.prefix = prefix
        self.suffix = suffix
        self.path = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix, suffix=self.suffix))
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path:
            safe_delete_temp_directory(self.path)
        return False  # 不处理异常


# 定义泛型类型
T = TypeVar('T')


class Result(BaseModel, Generic[T]):
    """
    统一API响应封装 - 增强版
    新增 success 字段，更直观表示请求状态
    """
    success: bool = Field(..., description="请求是否成功")
    code: int = Field(..., description="业务状态码")
    message: str = Field(..., description="响应消息")
    data: Optional[T] = Field(None, description="响应数据")
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp()),
                           description="响应时间戳(秒)")

    def to_response(self):
        """将 Result 对象转换为 FastAPI 响应"""
        return JSONResponse(
            content=self.dict(),
            status_code=200  # 所有响应使用200，实际状态通过code字段表示
        )

    @classmethod
    def ok(cls, data: T = None, message: str = "操作成功",
           code: int = 200) -> "Result[T]":
        """成功响应"""
        return cls(
            success=True,
            code=code,
            message=message,
            data=data
        )

    @classmethod
    def error(cls, code: int, message: str, data: T = None) -> "Result[T]":
        """错误响应"""
        return cls(
            success=False,
            code=code,
            message=message,
            data=data
        )

    @classmethod
    def not_found(cls, message: str = "资源未找到", data: T = None) -> "Result[None]":
        """资源未找到"""
        return cls.error(
            code=404,
            message=message,
            data=data
        )

    @classmethod
    def bad_request(cls, message: str = "请求参数错误", data: T = None) -> "Result[None]":
        """错误请求"""
        return cls.error(
            code=400,
            message=message,
            data=data
        )

    @classmethod
    def unauthorized(cls, message: str = "未授权访问", data: T = None) -> "Result[None]":
        """未授权"""
        return cls.error(
            code=401,
            message=message,
            data=data
        )

    @classmethod
    def forbidden(cls, message: str = "禁止访问", data: T = None) -> "Result[None]":
        """禁止访问"""
        return cls.error(
            code=403,
            message=message,
            data=data
        )

    @classmethod
    def internal_error(cls, message: str = "服务器内部错误", data: T = None) -> "Result[None]":
        """服务器内部错误"""
        return cls.error(
            code=500,
            message=message,
            data=data
        )

    @classmethod
    def paginated(
            cls,
            items: List[Any],
            total: int,
            page: int,
            page_size: int,
            message: str = "查询成功"
    ) -> "Result[Dict[str, Any]]":
        """分页响应"""
        pagination_data = {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
        return cls.ok(data=pagination_data, message=message)

    @classmethod
    def from_exception(cls, exc: Exception) -> "Result[None]":
        """从异常创建错误响应"""
        logger = logging.getLogger("uvicorn.error")
        logger.error(f"处理请求时出错: {str(exc)}", exc_info=True)

        return cls.error(
            code=500,
            message="服务器内部错误",
            data={"error_type": type(exc).__name__, "error_message": str(exc)}
        )

    @classmethod
    def validation_error(cls, errors: List[Dict]) -> "Result[Dict]":
        """验证错误响应"""
        error_details = []
        for error in errors:
            field = ".".join(str(loc) for loc in error["loc"])
            error_details.append({
                "field": field,
                "message": error["msg"],
                "type": error["type"]
            })

        return cls.error(
            code=422,
            message="请求参数验证失败",
            data={"errors": error_details}
        )


# 使用命名元组定义规范化的返回结构
class CommandResult(NamedTuple):
    success: bool
    returncode: int
    stdout: str
    stderr: str


async def run_command(cmd: List[str], timeout: float = None) -> CommandResult:
    """运行命令的异步辅助函数，含超时和错误处理"""
    try:
        # 创建子进程（增加shell=False的安全防护）
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            # 带超时的等待
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            # 超时处理：终止进程并返回特定错误
            process.kill()
            await process.wait()
            return CommandResult(
                success=False,
                returncode=-999,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds"
            )

        # 解码输出（增加错误处理）
        def safe_decode(data):
            return data.decode(errors="ignore").strip() if data else ""

        # 构造结果对象
        return CommandResult(
            success=process.returncode == 0,
            returncode=process.returncode,
            stdout=safe_decode(stdout),
            stderr=safe_decode(stderr)
        )

    except Exception as e:
        # 捕获所有执行异常
        return CommandResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr=f"Subprocess execution failed: {str(e)}"
        )


def get_bucket() -> oss.Bucket:
    """
        获取oss bucket
        :return:
    """

    os.environ['OSS_ACCESS_KEY_ID'] = 'xxx'
    os.environ['OSS_ACCESS_KEY_SECRET'] = 'xxx'
    ENDPOINT = 'oss-cn-hangzhou.aliyuncs.com'
    BUCKET = 'xxx'
    REGION = 'cn-hangzhou'

    auth = oss.ProviderAuth(EnvironmentVariableCredentialsProvider())

    return oss.Bucket(auth=auth, endpoint=ENDPOINT, bucket_name=BUCKET, region=REGION)


bucket = get_bucket()


async def download_file(url: str, save_path: str) -> None:
    """异步下载文件"""
    logger.info(f"开始下载文件: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"下载失败，状态码: {response.status}")

            with open(save_path, 'wb') as f:
                while True:
                    chunk = await response.content.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)


async def _download_or_use_local(video_path: str, output_path: Path) -> str:
    if video_path.startswith(('http://', 'https://')):
        # 生成唯一文件名防止覆盖
        ext = video_path.split('.')[-1].split('?')[0][:4]  # 获取扩展名
        filename = f"{uuid.uuid4().hex}.{ext}" if '.' in video_path else f"{uuid.uuid4().hex}.mp4"

        local_file = output_path / filename
        await download_file(video_path, str(local_file))
        if not local_file.exists():
            raise FileNotFoundError(f"下载文件失败: {video_path}")
        return str(local_file)
    else:
        local_file = Path(video_path)
        if not local_file.exists():
            raise FileNotFoundError(f"本地文件不存在: {video_path}")
        return str(local_file.resolve())  # 使用绝对路径


def get_duration(file_path: str) -> float:
    """获取音视频文件时长（秒）"""
    logger.info(f"开始获取文件时长: {file_path}")
    if file_path.endswith(('.mp4', '.mov', '.avi')):
        clip = VideoFileClip(file_path)
        duration = clip.duration
        clip.close()
        return duration
    elif file_path.endswith(('.wav', '.mp3')):
        clip = AudioFileClip(file_path)
        duration = clip.duration
        clip.close()
        return duration
    else:
        raise ValueError("不支持的文件格式")


@app.post("/merge_videos", response_model=Result, summary="合并视频文件")
async def merge_videos(
        video_paths: list[str],
        output_path: Optional[str] = None,
        merge_method: str = "concat"
) -> Result:
    """
    合并多个视频文件

    Args:
        video_paths: 视频文件路径列表
        output_path: 输出视频文件路径（可选）, 如果没有指定，则使用临时目录，程序运行结束后会自动删除
        merge_method: 合并方式（concat：简单拼接，filter：滤镜合并）

    Returns:
        合并结果信息
    """
    logger.info(f"开始合并视频文件: {video_paths}")
    # 创建工作目录
    output_path = create_directory(dir_path=output_path)
    try:
        paths = []
        if len(video_paths) < 2:
            return Result.bad_request(message="至少需要两个视频文件进行合并")

        # 判断 path 是否url
        for video_path in video_paths:
            local_path = await _download_or_use_local(video_path, output_path)
            paths.append(local_path)

        # 检查所有输入文件是否存在
        for path in paths:
            if not path or not os.path.exists(path):
                return Result.bad_request(message=f"错误：视频文件不存在 - {path}")

        # 设置输出路径
        first_file = Path(paths[0])
        output_file = first_file.parent / f"merged_video_{first_file.name}"
        logger.info(f"合并后的视频文件路径: {output_file}")
        if merge_method == "concat":
            # 创建临时文件列表
            list_file = output_path / "video_list.txt"
            with open(list_file, "w") as f:
                for path in paths:
                    f.write(f"file '{path}'\n")

            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                "-y",
                str(output_file)
            ]

            result = await run_command(cmd)

        else:  # filter方法
            # 构建复杂的filter命令
            inputs = []
            for path in paths:
                inputs.extend(["-i", path])

            filter_complex = ""
            for i in range(len(paths)):
                filter_complex += f"[{i}:v][{i}:a]"
            filter_complex += f"concat=n={len(paths)}:v=1:a=1[outv][outa]"

            cmd = [
                      "ffmpeg"
                  ] + inputs + [
                      "-filter_complex", filter_complex,
                      "-map", "[outv]",
                      "-map", "[outa]",
                      "-y",
                      str(output_file)
                  ]

            result = await run_command(cmd)

        if result.returncode == 0:
            logger.info(
                f"成功合并视频！\n输入文件: {', '.join(paths)}\n输出文件: {output_path}\n合并方式: {merge_method}")
            # 上传文件到oss
            first_file = Path(paths[0])
            file_key = f"test/merge_{first_file.name}"
            bucket.put_object_from_file(key=file_key, filename=str(output_file))
            logger.info(f"上传文件到oss成功！\n输出文件: {str(output_file)}")
            preview_url = bucket.sign_url('GET', file_key, 60 * 60 * 24 * 10)
            return Result.ok(data=preview_url)
        else:
            return Result.internal_error(message=f"合并失败：{result.stderr}")

    except Exception as e:
        logger.error(f"合并视频失败：{str(e)}")
        return Result.internal_error(message=f"合并视频失败：{str(e)}")
    finally:
        # 删除临时目录
        safe_delete_temp_directory(output_path)


@app.post("/merge_audios", response_model=Result, summary="合并音频文件")
async def merge_audios(
        audio_paths: list[str],
        output_path: Optional[str] = None,
        merge_method: str = "concat"
) -> Result:
    """
    合并多个音频文件

    Args:
        audio_paths: 音频文件路径列表
        output_path: 输出音频文件路径（可选），如果没有指定，则使用临时目录，程序运行结束后会自动删除
        merge_method: 合并方式（concat：拼接，mix：混音）

    Returns:
        合并结果信息
    """
    logger.info(f"开始合并音频文件: {audio_paths}")
    # 创建工作目录
    output_path = create_directory(dir_path=output_path)
    try:
        paths = []
        if len(audio_paths) < 2:
            return Result.bad_request(message="至少需要两个音频文件进行合并")

        # 判断 path 是否url
        for audio_path in audio_paths:
            local_path = await _download_or_use_local(audio_path, output_path)
            paths.append(local_path)

        # 检查所有输入文件是否存在
        for path in paths:
            if not os.path.exists(path):
                return Result.bad_request(message=f"错误：音频文件不存在 - {path}")

        first_file = Path(paths[0])
        output_file = first_file.parent / f"merged_audio.{first_file.suffix[1:]}"
        logger.info(f"合并后的音频文件路径: {output_file}")
        if merge_method == "concat":
            # 创建临时文件列表
            list_file = Path(output_path).parent / "audio_list.txt"
            with open(list_file, "w") as f:
                for path in paths:
                    f.write(f"file '{path}'\n")

            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                "-y",
                str(output_file)
            ]

            result = await run_command(cmd)

        else:  # mix方法
            inputs = []
            for path in paths:
                inputs.extend(["-i", path])

            filter_complex = f"amix=inputs={len(paths)}:duration=longest"

            cmd = [
                      "ffmpeg"
                  ] + inputs + [
                      "-filter_complex", filter_complex,
                      "-y",
                      str(output_file)
                  ]

            result = await run_command(cmd)

        if result.returncode == 0:
            logger.info(
                f"成功合并音频！\n输入文件: {', '.join(paths)}\n输出文件: {output_path}\n合并方式: {merge_method}")
            # 上传文件到oss
            first_file = Path(paths[0])
            file_key = f"test/merge_{first_file.name}"
            bucket.put_object_from_file(key=file_key, filename=str(output_file))
            logger.info(f"上传文件到oss成功！\n输出文件: {str(output_file)}")
            preview_url = bucket.sign_url('GET', file_key, 60 * 60 * 24 * 10)
            return Result.ok(data=preview_url)
        else:
            return Result.internal_error(message=f"合并失败：{result.stderr}")

    except Exception as e:
        return Result.internal_error(message=f"合并音频失败：{str(e)}")
    finally:
        # 删除临时目录
        safe_delete_temp_directory(output_path)


@app.post("/merge_video_and_audio", response_model=Result, summary="合并视频和音频文件")
async def merger_video_and_audio(
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        *,
        whisper_model: str = "turbo",
        subtitle_style: str = "FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF",
        video_loop: bool = True
) -> Result:
    """
    合并视频和音频文件，并添加字幕

    1. 下载远程文件（如果是URL）
    2. 合并视频和音频（如果音频比视频长则循环视频）
    3. 使用Whisper生成字幕
    4. 将字幕嵌入到视频中

    Args:
        video_path: 视频文件路径或URL
        audio_path: 音频文件路径或URL
        output_path: 输出文件路径（可选）, 如果没有指定，则使用临时目录，程序运行结束后会自动删除
        whisper_model: Whisper模型大小（base, small, medium等）
        subtitle_style: 字幕样式FFmpeg参数
        video_loop: 是否允许视频循环以匹配长音频

    Returns:
        最终视频文件路径
    """
    logger.info(f"开始合并视频和音频: video_path={video_path}, audio_path={audio_path}")
    # 创建临时工作目录
    output_path = create_directory(dir_path=output_path)
    try:

        # 处理视频文件（下载或使用本地文件）
        video_file = await _download_or_use_local(video_path=video_path, output_path=output_path)

        # 处理音频文件（下载或使用本地文件）
        audio_file = await _download_or_use_local(video_path=audio_path, output_path=output_path)

        # 步骤1: 合并视频和音频（处理音频长于视频的情况）

        # 创建视频循环（如果需要）
        video_loop_file = await create_video_loop(video_file, audio_file, video_loop, output_path)
        merged_video = output_path / f"merged_video_{Path(video_file).name}"

        # 合并循环视频和音频
        merge_cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-i', str(video_loop_file),
            '-i', str(audio_file),
            '-c:v', 'copy',  # 复制视频流
            '-c:a', 'aac',  # 转码音频为AAC
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest' if video_loop else '',  # 确保输出不超过最短的流
            str(merged_video)
        ]

        # 清理空参数
        merge_cmd = [arg for arg in merge_cmd if arg]

        # 运行FFmpeg合并命令
        proc = await run_command(merge_cmd)

        if proc.returncode != 0:
            raise Exception("视频音频合并失败")

        # 步骤2: 使用Whisper生成字幕
        # 生成SRT字幕
        srt_path = output_path / f"subtitle_{Path(audio_file).name}.srt"
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            subtitle_str = generate_subtitle(str(audio_file), whisper_model)
            srt_file.write(subtitle_str)

        # 步骤3: 将字幕嵌入到视频中（硬字幕）
        output_file = output_path / f"final_video_{Path(video_file).name}"
        subtitle_cmd = [
            'ffmpeg',
            '-y',
            '-i', str(merged_video),
            '-vf', f"subtitles={srt_path}:force_style='{subtitle_style}'",
            '-c:a', 'copy',
            str(output_file)
        ]

        # 运行FFmpeg字幕嵌入命令
        proc = await run_command(subtitle_cmd)
        if proc.returncode != 0:
            raise Exception("字幕嵌入失败")
        # 上传到oss
        file_key = f'test/{output_file.name}'
        bucket.put_object_from_file(key=file_key, filename=str(output_file))
        logger.info(f"上传成功: {file_key}")
        preview_url = bucket.sign_url('GET', file_key, 60 * 60 * 24 * 10)
        return Result.ok(data=preview_url)
    finally:
        # 删除临时目录
        safe_delete_temp_directory(output_path)


# 创建视频循环
async def create_video_loop(video_file: str, audio_file: str, video_loop: bool, output_path: Path) -> str:
    """创建视频循环"""
    logger.info(f"开始创建视频循环: video_file={video_file}, audio_file={audio_file}")
    # 获取视频和音频时长
    video_duration = get_duration(str(video_file))
    audio_duration = get_duration(str(audio_file))

    if video_loop and audio_duration > video_duration:
        # 计算需要循环的次数
        loop_count = math.ceil(audio_duration / video_duration)
        looped_video_file = str(output_path / f"looped_video_{Path(video_file).name}")
        # 创建视频循环
        loop_cmd = [
            'ffmpeg', '-y',
            '-stream_loop', str(loop_count),
            '-i', str(video_file),
            '-c:v', 'copy',
            '-t', str(audio_duration),  # 设置总时长为音频长度
            looped_video_file
        ]

        # 运行视频循环命令
        proc = await run_command(loop_cmd)

        if proc.returncode != 0:
            raise Exception("视频循环处理失败")

        # 使用循环后的视频进行合并
        video_file = looped_video_file
    return video_file


# 使用Whisper生成字幕
def generate_subtitle(audio_path: str, model_name: str = "turbo") -> str:
    """使用Whisper生成字幕"""
    logger.info(f"开始生成字幕: audio_path={audio_path}, model_name={model_name}")
    model = whisper.load_model(model_name)  # 使用turbo模型，速度较快
    result = model.transcribe(audio_path)

    # 生成SRT字幕
    output_blocks = []
    for i, segment in enumerate(result['segments'], 1):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()

        # 格式化时间戳 (HH:MM:SS,ms)
        start_hours = int(start // 3600)
        start_minutes = int((start % 3600) // 60)
        start_seconds = int(start % 60)
        start_millis = int((start - int(start)) * 1000)

        end_hours = int(end // 3600)
        end_minutes = int((end % 3600) // 60)
        end_seconds = int(end % 60)
        end_millis = int((end - int(end)) * 1000)

        start_str = f"{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d},{start_millis:03d}"
        end_str = f"{end_hours:02d}:{end_minutes:02d}:{end_seconds:02d},{end_millis:03d}"

        # 智能分行处理
        wrapped_text = intelligent_wrap(
            text=text,
            width=14,
            max_lines=10
        )
        # 重建字幕块
        output_blocks.append(f"{i}\n{start_str} --> {end_str}\n{wrapped_text}")

    return '\n\n'.join(output_blocks)


def intelligent_wrap(text, width, max_lines):
    """
    智能断行算法（保留词语完整性和标点规则）
    :param text: 原始文本
    :param width: 每行宽度
    :param max_lines: 最大行数
    """
    # 中文标点集合（在这些标点后允许换行）
    cjk_punctuation = '。！？，、；：”）》】.,!?;:'
    lines = []
    current_line = ""

    for char in text:
        # 尝试添加当前字符
        test_line = current_line + char

        # 检查是否超出宽度限制
        if len(test_line) >= width:
            # 查找最佳断点位置
            break_pos = find_break_position(current_line, cjk_punctuation)

            if break_pos > 0:
                # 找到合法断点
                lines.append(current_line[:break_pos + 1])
                current_line = current_line[break_pos + 1:] + char
            else:
                # 找不到合法断点时强制换行
                lines.append(current_line)
                current_line = char

            # 检查行数限制
            if len(lines) >= max_lines:
                # 行数超限时截断并添加省略号
                current_line = current_line[:width - 3] + "..." if len(current_line) > width - 3 else current_line
                break
        else:
            current_line = test_line

    # 添加最后一行
    if current_line and len(lines) < max_lines:
        lines.append(current_line)

    return '\n'.join(lines)


def find_break_position(line, delimiters):
    """
    在行中查找最佳断点位置
    :param line: 当前文本行
    :param delimiters: 允许断行的分隔符
    :return: 最佳断点位置索引
    """
    # 从行尾向前查找最近的合法分隔符
    for i in range(len(line) - 1, -1, -1):
        if line[i] in delimiters:
            return i

    # 查找最后一个空格（英文适用）
    last_space = line.rfind(' ')
    if last_space > len(line) * 0.8:  # 只在行尾附近空格处断开
        return last_space

    return -1  # 未找到合适断点


@app.post("/extract_video_frame", response_model=Result, summary="提取视频帧")
async def extract_video_frame(
        video_path: str,
        output_path: Optional[str] = None,
        frame_type: Literal["first", "last", "number", "time"] = "last",  # 使用字面量类型
        value: Optional[Union[int, float, str]] = None,
        quality: int = 2,  # 1-31 (1=最高质量)
        overwrite: bool = True,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe"
) -> Result:
    """
    提取视频指定帧为图片

    :param video_path: 输入视频路径
    :param output_path: 输出图片路径
    :param frame_type: 帧定位方式 (first, last, number, time)
    :param value: 帧号(int)或时间点(float秒/str时间码)
    :param quality: 输出图片质量 (1-31, 1=最佳)
    :param overwrite: 是否覆盖已存在文件
    :param ffmpeg_path: ffmpeg可执行文件路径
    :param ffprobe_path: ffprobe可执行文件路径
    """
    # 创建输出目录
    output_path = create_directory(dir_path=output_path)
    try:
        # 处理视频文件（下载或使用本地文件）
        video_file = await _download_or_use_local(video_path=video_path, output_path=output_path)
        # 验证输入文件
        if not os.path.isfile(video_file):
            raise FileNotFoundError(f"输入文件不存在: {video_file}")

        # 构建基础命令
        overwrite_flag = ["-y"] if overwrite else []
        quality_args = ["-q:v", str(quality)]

        output_file = output_path / f"frame_{frame_type}_{value}_{Path(video_file).stem}.jpg"

        # 处理不同帧类型
        if frame_type == "first":
            cmd = [
                ffmpeg_path,
                *overwrite_flag,
                "-ss", "0",  # 定位到起始
                "-i", video_file,
                "-vframes", "1",  # 只取1帧
                *quality_args,
                output_file
            ]

        elif frame_type == "last":
            cmd = [
                ffmpeg_path,
                *overwrite_flag,
                "-sseof", "-1",  # 定位到末尾前1秒
                "-i", video_file,
                "-update", "1",  # 更新模式
                "-vsync", "0",  # 禁止帧同步
                *quality_args,
                output_file
            ]

        elif frame_type == "number":
            if value is None or not isinstance(value, int) or value < 0:
                raise ValueError("使用'number'类型时需要提供非负整数的帧号")

            # 获取总帧数
            probe_cmd = [
                ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "default=nokey=1:nologger.info_wrappers=1",
                video_file
            ]

            try:
                total_frames = int(subprocess.check_output(probe_cmd).decode().strip())
            except Exception as e:
                raise RuntimeError(f"获取总帧数失败: {str(e)}")

            if value >= total_frames:
                raise ValueError(f"帧号超出范围 (最大: {total_frames - 1})")

            # 提取指定帧
            cmd = [
                ffmpeg_path,
                *overwrite_flag,
                "-i", video_file,
                "-vf", f"select=eq(n\\,{value})",  # 注意转义逗号
                "-vframes", "1",
                *quality_args,
                output_file
            ]

        elif frame_type == "time":
            if value is None:
                raise ValueError("使用'time'类型时需要提供时间值")

            time_arg = str(value) if isinstance(value, (int, float)) else value

            cmd = [
                ffmpeg_path,
                *overwrite_flag,
                "-ss", time_arg,  # 定位时间点
                "-i", video_file,
                "-vframes", "1",  # 只取1帧
                *quality_args,
                output_file
            ]

        else:
            # 由于使用了字面量类型，理论上不会执行到这里
            raise ValueError(f"不支持的帧类型: {frame_type}。可选: 'first', 'last', 'number', 'time'")

        # 执行FFmpeg命令

        result = await run_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg执行失败: {result.stderr}")

        # 上传到oss
        file_key = f'test/{output_file.name}'
        bucket.put_object_from_file(key=file_key, filename=str(output_file))
        logger.info(f"上传成功: {file_key}")
        preview_url = bucket.sign_url('GET', file_key, 60 * 60 * 24 * 10)
        return Result.ok(data=preview_url)

    finally:
        # 删除临时文件
        safe_delete_temp_directory(path=output_path)


# 使用示例
async def main():
    try:
        # 示例1: 使用本地文件
        result = await merger_video_and_audio(
            video_path="/Users/myproject/ffmpeg_python_mcp/test/merged_video.mp4",
            audio_path="/Users/Downloads/test.wav",
            output_path="final_output.mp4"
        )
        logger.info(f"处理完成: {result}")

        # # 示例2: 使用远程文件
        # result = await merger_video_and_audio(
        #     video_path="https://example.com/short_video.mp4",
        #     audio_path="https://example.com/long_audio.wav"
        # )
        # logger.info(f"处理完成: {result}")
        #
        # # 示例3: 禁用视频循环
        # result = await merger_video_and_audio(
        #     video_path="long_video.mp4",
        #     audio_path="short_audio.wav",
        #     video_loop=False
        # )
        logger.info(f"处理完成: {result}")

    except Exception as e:
        logger.info(f"处理失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8899)
