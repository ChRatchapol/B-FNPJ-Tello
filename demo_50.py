# Final Project Demo @ 50% Progress

# ~ This python file use Black as formatter.
# <3 For the best experience, use the 'Better Comments' extension in VS Code
# <3 and settings in https://github.com/ChRatchapol/BetterCommentsSetting

# ## Todo List for Testing
# #- todo list for state and log
# -todo 1. get state
# -todo 2. create state log file
# -todo 3. create log file
# #- todo list for image
# -todo 1. get image stream to GCS in openCV
# -todo 2. color detection
# -todo 3. get direction from image proc.
# #- todo list for control
# -todo 1. control drone via command
# -todo 2. control drone from data provide from color detection

# | IMPORT SECTION
import cv2  # ? GStremer is needed to run this file. (Environment: OS = Windows11, Python = 3.7.9, OpenCV = 4.4.0 (with extra), GStreamer = 1.16.3 (msvc-x86_64))
import numpy as np
import os
import socket
import sys
import threading
import time

from datetime import datetime, timedelta
from multiprocessing import Process, Queue
from queue import Empty, Full
from sympy import symbols, Eq, solve
from typing import Any, List, Tuple, Union

# | PREDEFINE SECTION
FORMAT = "utf-8"  # format for encode/decode string from/to Tello
IP = ""  # any IP
CMD_PORT = 8889  # command port for GCS
STT_PORT = 8890  # state port for GCS
IMG_PORT = 11111  # image port for GCS
TELLO_IP = "192.168.10.1"  # Tello IP (Static)
CMD_SOCK = None  # command socket init.
TERMINATE = False  # terminate flag for threads (in case of daemon don't work)

# ? ↓ These values might be sensors size because when Tello switches to video mode in the Tello application, The video crop a little.
MAX_VIDEO_WIDTH = 960  # max video width Tello can provide
MAX_VIDEO_HEIGHT = 720  # max video height Tello can provide
FPS = 25  # fps (30 in Tello specs. but we'll use 25)

# * color detection stuffs
LOWER_RED = np.array([0, 145, 50])  # lower red color for color detection
UPPER_RED = np.array([5, 255, 255])  # upper red color for color detection
LOWER_GRN = np.array([62, 54, 52])  # lower green color for color detection
UPPER_GRN = np.array([86, 255, 241])  # upper green color for color detection
# LOWER_MGT = np.array([117, 184, 62])  # lower magenta color for color detection
# UPPER_MGT = np.array([155, 229, 255])  # upper magenta color for color detection
LOWER_MGT = np.array([144, 139, 70])  # lower magenta color for color detection
UPPER_MGT = np.array([179, 255, 255])  # upper magenta color for color detection
# RGB collor predefine
BLU = (0, 0, 255)
CYN = (0, 255, 255)
GRN = (0, 255, 0)
MGT = (255, 0, 255)
RED = (255, 0, 0)
YLE = (255, 255, 0)
PADDING = 20  # padding for bounding box

# * zoning coord
zone_size = 3.75
tl_z = (
    (0, 0),
    (int((MAX_VIDEO_WIDTH - 1) // zone_size), int((MAX_VIDEO_HEIGHT - 1) // zone_size)),
)
tm_z = (
    (int((MAX_VIDEO_WIDTH - 1) // zone_size), 0),
    (
        int((MAX_VIDEO_WIDTH - 1) - ((MAX_VIDEO_WIDTH - 1) // zone_size)),
        int((MAX_VIDEO_HEIGHT - 1) // zone_size),
    ),
)
tr_z = (
    ((MAX_VIDEO_WIDTH - 1) - int((MAX_VIDEO_WIDTH - 1) // zone_size), 0),
    ((MAX_VIDEO_WIDTH - 1), int((MAX_VIDEO_HEIGHT - 1) // zone_size)),
)
ml_z = (
    (0, int((MAX_VIDEO_HEIGHT - 1) // zone_size)),
    (
        int((MAX_VIDEO_WIDTH - 1) // zone_size),
        (MAX_VIDEO_HEIGHT - 1) - int((MAX_VIDEO_HEIGHT - 1) // zone_size),
    ),
)
mc_z = (
    (int((MAX_VIDEO_WIDTH - 1) // zone_size), int((MAX_VIDEO_HEIGHT - 1) // zone_size)),
    (
        (MAX_VIDEO_WIDTH - 1) - int((MAX_VIDEO_WIDTH - 1) // zone_size),
        (MAX_VIDEO_HEIGHT - 1) - int((MAX_VIDEO_HEIGHT - 1) // zone_size),
    ),
)
mr_z = (
    (
        (MAX_VIDEO_WIDTH - 1) - int((MAX_VIDEO_WIDTH - 1) // zone_size),
        int((MAX_VIDEO_HEIGHT - 1) // zone_size),
    ),
    (
        (MAX_VIDEO_WIDTH - 1),
        (MAX_VIDEO_HEIGHT - 1) - int((MAX_VIDEO_HEIGHT - 1) // zone_size),
    ),
)
bl_z = (
    (0, (MAX_VIDEO_HEIGHT - 1) - int((MAX_VIDEO_HEIGHT - 1) // zone_size)),
    (int((MAX_VIDEO_WIDTH - 1) // zone_size), (MAX_VIDEO_HEIGHT - 1)),
)
bm_z = (
    (
        int((MAX_VIDEO_WIDTH - 1) // zone_size),
        (MAX_VIDEO_HEIGHT - 1) - int((MAX_VIDEO_HEIGHT - 1) // zone_size),
    ),
    (
        (MAX_VIDEO_WIDTH - 1) - int((MAX_VIDEO_WIDTH - 1) // zone_size),
        (MAX_VIDEO_HEIGHT - 1),
    ),
)
br_z = (
    (
        (MAX_VIDEO_WIDTH - 1) - int((MAX_VIDEO_WIDTH - 1) // zone_size),
        (MAX_VIDEO_HEIGHT - 1) - int((MAX_VIDEO_HEIGHT - 1) // zone_size),
    ),
    ((MAX_VIDEO_WIDTH - 1), (MAX_VIDEO_HEIGHT - 1)),
)

# * terminal output stuffs
# ? https://en.wikipedia.org/wiki/ANSI_escape_code
ESC = "\x1B"
CSI = f"{ESC}["

# | CLASS SECTION


class Q:
    def __init__(self, size: int = -1) -> None:
        self.size = size
        self._queue = []
        self.Full = False
        self.Empty = True

    def put(self, item: Any) -> None:
        if not self.Full:
            self._queue.append(item)
            self.updateStatus()
        else:
            raise Q.FULL("The queue is full.")

    def get(self) -> Any:
        if not self.Empty:
            res = self._queue.pop(0)
            self.updateStatus()
            return res
        else:
            raise Q.EMPTY("The queue is empty.")

    def updateStatus(self) -> None:
        if len(self._queue) == 0:
            self.Empty = True
            self.Full = False
        elif len(self._queue) == self.size:
            self.Empty = False
            self.Full = True
        else:
            self.Empty = False
            self.Full = False

    class FULL(Exception):
        pass

    class EMPTY(Exception):
        pass


# | FUNCTION SECTION
def cprint(
    text: str,
    color: Union[Tuple[int, int, int], str] = "white",
    end: str = "\n",
    sep: str = " ",
    file=sys.stdout,
    flush: bool = False,
) -> None:
    """
    print text in color base on ASCII escape code

    Parameters
    ----------
    text : str
        string that will be printed
    color : Tuple[int, int, int] | str, optional
        string color, can be a tuple of 3 integers (0-255) represents red, green, and blue or color name as string, by default "white"
        color name : "black", "blue", "cyan", "green", "magenta", "red", "yellow", "white"
    end : str, optional
        string inserted between values, by default a space (" ")
    sep : str, optional
        string appended after the last value, by default a newline ("\\n")
    file, optional
        a file-like object (stream), by default current sys.stdout
    flush : bool, optional
        whether to forcibly flush the stream, by default False

    Raise
    ------
    KeyError
        Raise when an invalid color name receives.
    """
    global CSI

    # * color
    preset_color = {
        "black": (0, 0, 0),
        "blue": (0, 0, 255),
        "cyan": (0, 255, 255),
        "green": (0, 255, 0),
        "magenta": (255, 0, 255),
        "red": (255, 0, 0),
        "yellow": (255, 255, 0),
        "white": (255, 255, 255),
    }
    if type(color) is str:
        try:
            color = preset_color[color]
        except KeyError:
            raise KeyError("Invlaid color!")

    r = color[0]
    g = color[1]
    b = color[2]

    print(
        f"{CSI}38;2;{r};{g};{b}m{text}{CSI}m", end=end, file=file, sep=sep, flush=flush
    )


def recvThread() -> None:
    """
    Thread for receiving command response from Tello.
    """
    global CMD_SOCK  # command socket from global
    global TERMINATE  # terminate flag from global
    global CMD_LOG_FILE_NAME  # log file name
    global CMD_LOG_Q

    if not os.path.isdir("log"):
        os.mkdir("log")

    while not TERMINATE:  # receiving loop
        try:
            data, server = CMD_SOCK.recvfrom(1518)  # receive response from socket
        except BlockingIOError:  # ignore i/o error
            continue
        # cprint(
        #     data.decode(encoding=FORMAT).strip(), "magenta"
        # )  # print response output (for now)
        CMD_LOG_Q.put(
            (
                datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
                "recv",
                data.decode(encoding=FORMAT).strip(),
            )
        )


def stateThread(rate: int = 2) -> None:
    """
    Thread for receiving state from Tello.

    Parameters
    ----------
    rate : int, optional
        receiving rate, by default 2
    """
    global TERMINATE  # terminate flag from global
    global STATE  # STATE var.
    global STT_LOG_Q

    ADDR = (IP, STT_PORT)  # address for state socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # create socket
    sock.bind(ADDR)  # bind socket to state address

    start = datetime.now()  # start timer
    while not TERMINATE:  # receiving loop
        # ↓ get state from socket (no need to ignore i/o error because this socket is only read and not fast)
        data, server = sock.recvfrom(1518)
        if datetime.now() - start >= timedelta(seconds=1 / rate):  # wait for timer
            state = (
                data.decode(encoding=FORMAT).strip().split(";")
            )  # slice each value to list
            STATE = {
                i.split(":")[0]: float(i.split(":")[1])
                if i.split(":")[0] in ["baro", "agx", "agy", "agz"]
                else int(i.split(":")[1])
                for i in state
                if i != ""
            }  # convert state list to STATE dict
            STT_LOG_Q.put((datetime.now().strftime("%d/%m/%Y-%H:%M:%S"), STATE))
            start = datetime.now()  # reset timer


def imageThread(cap_q: Queue) -> None:
    """
    Thread for receiving image from Tello.
    """
    IMG_PORT = 11111
    MAX_VIDEO_WIDTH = 960
    MAX_VIDEO_HEIGHT = 720
    FPS = 25
    ESC = "\x1B"
    CSI = f"{ESC}["

    # cap = cv2.VideoCapture(f"udp://@0.0.0.0:{IMG_PORT}") # get video feed from Tello by pure OpenCV (1-2 seconds delay)

    # ↓ get video feed from Tello by Gstreamer + OpenCV (less than 1 second delay)
    pipeline = f"udpsrc port={IMG_PORT} ! video/x-h264,stream-format=byte-stream,skip-first-bytes=2,width={MAX_VIDEO_WIDTH},height={MAX_VIDEO_HEIGHT},framerate={FPS}/1 ! queue ! decodebin ! videoconvert ! appsink"
    # ? ↑ gstreamer pipeline for tello ref: https://github.com/Ragnar-H/TelloGo

    cap = cv2.VideoCapture(
        pipeline, cv2.CAP_GSTREAMER
    )  # create VideoCapture for video feed

    time.sleep(1)

    print(
        f"{CSI+'F'}{CSI+'2K'}{CSI+'38;2;0;255;0m'}VideoCapture created, go in video feed loop{CSI+'m'}"
    )
    while 1:
        ret, frame = cap.read()
        IMAGE = (ret, frame)

        try:
            cap_q.put_nowait(IMAGE)
        except Full:
            try:
                cap_q.get_nowait()
            except Empty:
                pass
            # cap_q.put_nowait(IMAGE)

    cap.release()  # release VideoCapture


def imageProc(q: Queue) -> None:
    """
    image process

    Parameters
    ----------
    q : Queue
        The queue for transmit data between from image proc. and main proc.
    """
    FPS = 15
    MAX_VIDEO_WIDTH = 960
    MAX_VIDEO_HEIGHT = 720

    IMAGE = (False, None)
    cap_q = Queue(1)
    img_thrd = threading.Thread(target=imageThread, args=([cap_q]), daemon=True)
    img_thrd.start()

    START = time.time()
    queue = []
    TERMINATE = False
    FOUND = False
    LAST_FOUND = time.time()
    while 1:
        try:
            TMP = cap_q.get_nowait()
        except Empty:
            pass
        else:
            IMAGE = TMP

        ret, frame = IMAGE  # get frame

        cen = (-1, -1)

        if ret:  # check if the frame is valid
            res = frame.copy()
            frame = cv2.GaussianBlur(frame, (3, 3), 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_MGT, UPPER_MGT)
            mask = cv2.GaussianBlur(mask, (9, 9), 2)
            ret, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_TOZERO)

            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            largest_cnt = None
            largest_area = -1

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 3000 and area > largest_area:
                    largest_cnt = cnt
                    largest_area = area

            if largest_cnt is not None:
                FOUND = True
                LAST_FOUND = time.time()
                peri = cv2.arcLength(largest_cnt, True)
                approx = cv2.approxPolyDP(largest_cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cen = getCenter(x, y, w, h, PADDING)
                cv2.circle(res, cen, 5, (255, 255, 0), -1)

                cv2.drawContours(res, largest_cnt, -1, CYN, 7)
                peri = cv2.arcLength(largest_cnt, True)
                approx = cv2.approxPolyDP(largest_cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(
                    res,
                    (x - PADDING, y - PADDING),
                    (x + w + PADDING, y + h + PADDING),
                    MGT,
                    5,
                )

                cv2.putText(
                    res,
                    f"Area: {largest_area:,.0f}",
                    (x + w + PADDING + 20, y + PADDING),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    MGT,
                    3,
                )
            elif FOUND and (time.time() - LAST_FOUND >= 5):
                TERMINATE = True
                try:
                    q.put_nowait((TERMINATE, None))
                except Full:
                    try:
                        q.get_nowait()
                    except Empty:
                        pass
                    # q.put_nowait((TERMINATE, None))
                CMD_LOG_Q.put(
                    (
                        datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
                        "IMAGE",
                        "Lost",
                    )
                )
                cprint("LOST", "red")
                break

            direction = Zone(res, cen)
            if largest_area <= 10000 and largest_cnt is not None:
                if direction == "":
                    direction = "forward"
                else:
                    direction += "-forward"
            elif largest_area >= 20000 and largest_cnt is not None:
                if direction == "":
                    direction = "back"
                else:
                    direction += "-back"

            try:
                q.put_nowait((direction, cen))
            except Full:
                try:
                    q.get_nowait()
                except Empty:
                    pass
                # q.put_nowait(direction)

            if time.time() - START >= round(1 / FPS, 2):
                queue.append(res)
                START = time.time()

            cv2.imshow("feed", res)  # show frame
            # cv2.imshow("mask", mask)

            key = cv2.waitKey(1) & 0xFF  # get key pressed

            if key == ord("q"):  # if 'q' is pressed quit the loop
                TERMINATE = True
                try:
                    q.put_nowait((TERMINATE, None))
                except Full:
                    try:
                        q.get_nowait()
                    except Empty:
                        pass
                    # q.put_nowait((TERMINATE, None))
                break

    cv2.destroyAllWindows()

    if not os.path.isdir("video"):
        os.mkdir("video")

    result = cv2.VideoWriter(
        f"./video/vid_{datetime.now().strftime('%d%m%Y-%H%M%S')}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (MAX_VIDEO_WIDTH, MAX_VIDEO_HEIGHT),
    )

    for frame in queue:
        result.write(frame)

    result.release()
    cprint("video saved", "cyan")

    for _ in range(10):
        try:
            q.put_nowait((TERMINATE, None))
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass

            try:
                q.put_nowait((TERMINATE, None))
            except Full:
                continue

    cprint("end of image proc.", "red")


def Zone(frame: np.ndarray, point: Tuple[int, int]) -> str:
    """
    draw zone on image and return direction the drone should go

    Parameters
    ----------
    frame : np.ndarray
        the frame that will draw zones on
    point : Tuple[int, int]
        point of object

    Returns
    -------
    str
        direction drone should go to make the point in center zone
    """
    global tl_z
    global tm_z
    global tr_z
    global ml_z
    global mc_z
    global mr_z
    global bl_z
    global bm_z
    global br_z

    rgb2bgr = lambda color: color[::-1]

    direction = ""

    zones = [tl_z, tm_z, tr_z, ml_z, mc_z, mr_z, bl_z, bm_z, br_z]
    # label = [
    #     "up-left",
    #     "up",
    #     "up-right",
    #     "left",
    #     "",
    #     "right",
    #     "down-left",
    #     "down",
    #     "down-right",
    # ]
    label = [
        "up-ccw",
        "up",
        "up-cw",
        "ccw",
        "",
        "cw",
        "down-ccw",
        "down",
        "down-cw",
    ]

    highlighted_zone = None
    for zone in zones:
        if inZone(point, zone):
            highlighted_zone = zone

    for zone in zones:
        if zone == highlighted_zone:
            continue
        cv2.rectangle(frame, zone[0], zone[1], rgb2bgr(MGT), 1)

    if highlighted_zone != None:
        if highlighted_zone == mc_z:
            cv2.rectangle(
                frame, highlighted_zone[0], highlighted_zone[1], rgb2bgr(GRN), 3
            )
        else:
            cv2.rectangle(
                frame, highlighted_zone[0], highlighted_zone[1], rgb2bgr(RED), 3
            )
        direction = label[zones.index(highlighted_zone)]

    if point == (-1, -1):
        direction = ""

    return direction


def inZone(
    point: Tuple[int, int], zone: Tuple[Tuple[int, int], Tuple[int, int]]
) -> bool:
    """
    Determine whether the point is in which zone or not.

    Parameters
    ----------
    point : Tuple[int, int]
        point represents object
    zone : Tuple[Tuple[int, int], Tuple[int, int]]
        coord of zone (top left and bottom right corner)

    Returns
    -------
    bool
        True if the point is in the zone, False if not.
    """
    tl_z = zone[0]
    br_z = zone[1]

    if (point[0] >= tl_z[0] and point[0] <= br_z[0]) and (
        point[1] >= tl_z[1] and point[1] <= br_z[1]
    ):
        return True
    else:
        return False


def getCenter(
    x: int, y: int, w: int, h: int, padding: int = PADDING
) -> Tuple[int, int]:
    """
    find center of rectangle given top left-corner coordinate, width, and height with optional padding

    Parameters
    ----------
    x : int
        x value of top-left coord.
    y : int
        y value of top-left coord.
    w : int
        width of rectangle
    h : int
        height of rectangle
    padding : int, optional
        additional padding space of the rectangle, by default PADDING (global)

    Returns
    -------
    Tuple[int, int]
        The tuple of 2 integers represents coord. of center of the rectangle.
    """
    tl = (x - padding, y - padding)
    tr = (x + w + padding, y - padding)
    bl = (x - padding, y + h + padding)
    br = (x + w + padding, y + h + padding)

    x, y = symbols("x y")

    m1 = round((tl[1] - br[1]) / (tl[0] - br[0]), 3)
    eq1 = Eq(y - tl[1], m1 * (x - tl[0]))

    m2 = round((tr[1] - bl[1]) / (tr[0] - bl[0]), 3)
    eq2 = Eq(y - tr[1], m2 * (x - tr[0]))

    cen = solve((eq1, eq2), (x, y), simplify=False, rational=False)
    cen = (round(float(cen[x])), round(float(cen[y])))
    return cen


def matrixIndex(mat: List[List[Any]], item: Any) -> Tuple[int, int]:
    """
    find index of item from 2-dimension matrix

    Parameters
    ----------
    mat : List[List[Any]]
        matrix
    item : Any
        item that want to get index of

    Returns
    -------
    Tuple[int, int]
        index of the item in matrix if item is not found (-1,-1) is returned
    """
    row = [i for i in mat if item in i]
    if row == []:
        return (-1, -1)
    else:
        return (mat.index(row[0]), row[0].index(item))


def cmd(direction_str: str) -> None:
    """
    sent command to make Tello fly with direction provide

    Parameters
    ----------
    direction_str : str
        direction drone will fly
    """
    global CMD_SOCK
    global STATE

    _zone_dct = {
        "up-ccw": "tl_z",
        "up": "tm_z",
        "up-cw": "tr_z",
        "ccw": "ml_z",
        "": "mc_z",
        "cw": "mr_z",
        "down-ccw": "bl_z",
        "down": "bm_z",
        "down-cw": "br_z",
    }

    _zone_mat = [
        ["tl_z", "tm_z", "tr_z"],
        ["ml_z", "mc_z", "mr_z"],
        ["bl_z", "bm_z", "br_z"],
    ]

    if direction_str == "":
        return

    directions = direction_str.split("-")
    current_zone = _zone_dct[
        "-".join([s for s in direction_str.split("-") if s not in ["forward", "back"]])
    ]

    target_zone = ""
    for direction in directions:
        if direction in ["ccw", "cw"]:
            amp = "15"
        elif direction in ["forward", "back"]:
            amp = "23"
        else:
            amp = "20"

        if direction == "up":
            row, col = matrixIndex(_zone_mat, current_zone)
            target_zone = _zone_mat[row + 1][col]
        elif direction == "down":
            row, col = matrixIndex(_zone_mat, current_zone)
            target_zone = _zone_mat[row - 1][col]
        elif direction == "ccw":
            row, col = matrixIndex(_zone_mat, current_zone)
            target_zone = _zone_mat[row][col + 1]
        elif direction == "cw":
            row, col = matrixIndex(_zone_mat, current_zone)
            target_zone = _zone_mat[row][col - 1]
        else:
            target_zone = current_zone

        msg = f"{direction} {amp}".encode(encoding=FORMAT)  # create message

        START = time.time()
        while time.time() - START <= 3:
            try:
                (
                    new_direction,
                    point,
                ) = (
                    q.get_nowait()
                )  # get direction and center of object from image proc.
                if (new_direction != direction_str) or (new_direction is True):
                    return
            except Empty:
                continue

            sent = socketSend(
                CMD_SOCK, msg, (TELLO_IP, CMD_PORT)
            )  # send message to Tello
            if (direction not in ["forward", "back"]) and inZone(
                point, globals()[target_zone]
            ):
                msg = f"forward 0".encode(encoding=FORMAT)  # create message
                sent = socketSend(
                    CMD_SOCK, msg, (TELLO_IP, CMD_PORT)
                )  # send message to Tello
                time.sleep(0.5)
                break
            time.sleep(0.1)


def socketSend(sock: socket.socket, msg: bytes, addr: Tuple[str, int]) -> int:
    global CMD_LOG_FILE_NAME  # log file name
    global CMD_LOG_Q

    sent = sock.sendto(msg, addr)  # send message to Tello
    CMD_LOG_Q.put(
        (
            datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
            "send",
            msg.decode(encoding=FORMAT),
        )
    )
    return sent


def cmdLogWrite() -> None:
    global CMD_LOG_FILE_NAME  # log file name
    global CMD_LOG_Q
    global CMD_LOG_FILE
    global LOG_TIME
    global TERMINATE

    _sep = "\t"
    while not TERMINATE or not CMD_LOG_Q.Empty:
        try:
            _time, _type, _data = CMD_LOG_Q.get()
        except Q.EMPTY:
            continue
        CMD_LOG_FILE.write(f"{_time}{_sep}{_type}{_sep}>>>{_sep}{_data}\n")


def sttLogWrite() -> None:
    global STT_LOG_FILE_NAME  # log file name
    global STT_LOG_Q
    global LOG_TIME
    global TERMINATE

    def dct2str(dct: dict) -> str:
        keys = dct.keys()
        lst = [f"{key}\t:\t{dct[key]}" for key in keys]
        return "\t".join(lst)

    _sep = "\t"
    while not TERMINATE or not STT_LOG_Q.Empty:
        try:
            _time, _state = STT_LOG_Q.get()
        except Q.EMPTY:
            continue
        STT_LOG_FILE.write(f"{_time}{_sep}{dct2str(_state)}\n")


# | MAIN CODE
if __name__ == "__main__":
    if not os.path.isdir("log"):
        os.mkdir("log")

    LOG_TIME = datetime.now().strftime("%d%m%Y-%H%M%S")
    if not os.path.isdir(f"log/{LOG_TIME}"):
        os.mkdir(f"log/{LOG_TIME}")

    if not os.path.isdir("output"):
        os.mkdir("output")

    LOG_TIME = datetime.now().strftime("%d%m%Y-%H%M%S")

    CMD_LOG_FILE_NAME = "flight_command.log"
    CMD_LOG_Q = Q()

    CMD_LOG_FILE = open(f"./log/{LOG_TIME}/{CMD_LOG_FILE_NAME}", "wt")
    CMD_LOG_FILE.write(
        f"FLIGHT COMMAND LOG START AT {datetime.now().strftime('%d/%m/%Y-%H:%M:%S')}\n{'='*20}\n"
    )

    cmd_log_thrd = threading.Thread(target=cmdLogWrite)
    cmd_log_thrd.start()

    STT_LOG_FILE_NAME = f"flight_state.log"
    STT_LOG_Q = Q()

    STT_LOG_FILE = open(f"./log/{LOG_TIME}/{STT_LOG_FILE_NAME}", "wt")
    STT_LOG_FILE.write(
        f"FLIGHT STATE LOG START AT {datetime.now().strftime('%d/%m/%Y-%H:%M:%S')}\n{'='*20}\n"
    )

    stt_log_thrd = threading.Thread(target=sttLogWrite)
    stt_log_thrd.start()

    GCS_CMD_ADDR = (IP, 9000)  # command address for GCS

    CMD_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # create socket
    CMD_SOCK.bind(GCS_CMD_ADDR)  # bind socket to command address for GCS
    CMD_SOCK.setblocking(False)  # won't wait (timeout = 0)
    cprint("socket created", "green")

    # ↓ create receiving thread for command response
    recv_thd = threading.Thread(target=recvThread, daemon=True)
    recv_thd.start()  # start receiving thread for command response
    cprint("command response thread started", "green")

    msg = "command".encode(encoding=FORMAT)  # create message (change to SDK mode)
    sent = socketSend(CMD_SOCK, msg, (TELLO_IP, CMD_PORT))  # send message to Tello
    msg = "streamon".encode(encoding=FORMAT)  # create message (open video stream)
    sent = socketSend(CMD_SOCK, msg, (TELLO_IP, CMD_PORT))  # send message to Tello
    msg = "battery?".encode(encoding=FORMAT)  # create message (get battery info.)
    sent = socketSend(CMD_SOCK, msg, (TELLO_IP, CMD_PORT))  # send message to Tello
    msg = "speed 85".encode(encoding=FORMAT)  # create message (get battery info.)
    sent = socketSend(CMD_SOCK, msg, (TELLO_IP, CMD_PORT))  # send message to Tello

    time.sleep(1)  # wait for everything to settle

    STATE = {}
    stt_thrd = threading.Thread(
        target=stateThread, args=tuple([4]), daemon=True
    )  # create state receiving thread
    stt_thrd.start()  # start state receiving thread

    q = Queue(1)
    image_proc = Process(target=imageProc, args=([q]))
    image_proc.start()

    time.sleep(1)  # wait for everything to settle

    # msg = "takeoff".encode(encoding=FORMAT)  # create message (takeoff)
    # sent = socketSend(CMD_SOCK, msg, (TELLO_IP, CMD_PORT))  # send message to Tello

    time.sleep(3)

    while not TERMINATE:  # control loop
        try:
            (
                direction,
                center,
            ) = q.get_nowait()  # get direction and center of object from image proc.
            if direction == "":
                continue
        except Empty:
            continue

        if (
            direction is True
        ) or not image_proc.is_alive():  # if get terminate signal from image proc.
            TERMINATE = True  # set TERMINATE flag to True to terminate all threads
            cprint("got TERMINATE", "red")
            break

        cmd(direction)  # control direction

    # msg = "land".encode(encoding=FORMAT)  # create message (land)
    # sent = socketSend(CMD_SOCK, msg, (TELLO_IP, CMD_PORT))  # send message to Tello

    print()

    CMD_LOG_FILE.write(
        f"{'='*20}\nFLIGHT COMMAND LOG END AT {datetime.now().strftime('%d/%m/%Y-%H:%M:%S')}"
    )
    CMD_LOG_FILE.close()

    STT_LOG_FILE.write(
        f"{'='*20}\nFLIGHT STATE LOG END AT {datetime.now().strftime('%d/%m/%Y-%H:%M:%S')}"
    )
    STT_LOG_FILE.close()

    cprint("end", "red")
