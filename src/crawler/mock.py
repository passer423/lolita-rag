"""离线 mock 语料。

作用有两个：
1. 没有微博 Cookie 时也能跑通「采集 → 清洗 → 切分 → 向量化 → 检索 → 生成」全链路；
2. 作为测试夹具，检索质量回归时不依赖网络。

语料内容刻意混入 emoji、#话题#、@提及、短链、转发前缀等噪声，
用来验证 ``preprocessing.clean`` 的清洗效果；最后两条是与 Lolita 无关的干扰项，
用来观察检索的区分度。
"""

from __future__ import annotations

from src.schema import Post

MOCK_POSTS: list[dict] = [
    {
        "id": "mock-001",
        "text": "新手求问：JSK 和 OP 到底怎么选？我理解 JSK 是吊带裙要自己配内搭，OP 是一体式有袖子的。"
        "想入坑第一条的话，是不是 OP 更省事？#Lolita##新手入坑#",
        "author": "裙摆与猫",
        "created_at": "2026-03-02 21:14",
        "likes": 328,
        "comments": 76,
        "reposts": 12,
        "url": "https://m.weibo.cn/detail/mock-001",
    },
    {
        "id": "mock-002",
        "text": "接上条，补充一下：JSK 的搭配自由度最高，内搭衬衫、泡泡袖、无袖都可以换，"
        "一条 JSK 能穿出好几种感觉；OP 更省心但季节适应性差一些，夏天穿长袖 OP 会很热。"
        "预算有限的话我会先买 JSK，配两件不同色的内搭就够了。",
        "author": "裙摆与猫",
        "created_at": "2026-03-02 21:31",
        "likes": 201,
        "comments": 33,
        "reposts": 8,
    },
    {
        "id": "mock-003",
        "text": "终于收到心心念念的柄图了！这条的柄是茶壶和草莓塔，配色偏奶蓝，"
        "实物质感比官图好太多。就是腰围有点紧，建议姐妹们按官方尺码表再放 2cm 余量下单。",
        "author": "茶会迟到选手",
        "created_at": "2026-02-27 19:02",
        "likes": 512,
        "comments": 94,
        "reposts": 21,
    },
    {
        "id": "mock-004",
        "text": "关于国牌和日牌的价格差，说点实话：日牌本体贵，但二手流通性好，"
        "穿一季不喜欢了基本能回血七成；国牌便宜、上新快，但版型和做工波动大，"
        "同一家店的款可能一条封神一条翻车。预算 500 以内我建议先看国牌现货，别急着冲日牌预售。",
        "author": "理智剁手日记",
        "created_at": "2026-02-25 10:48",
        "likes": 866,
        "comments": 158,
        "reposts": 143,
    },
    {
        "id": "mock-005",
        "text": "第一次去茶会的礼仪整理：①提前确认 dress code，多数是 formal 或 semi-formal；"
        "②不要穿和别人撞柄的裙子，撞柄会很尴尬；③拍照前先问对方是否愿意出镜；"
        "④甜品一般可以打包但别整盘端走。祝大家玩得开心～",
        "author": "茶会迟到选手",
        "created_at": "2026-02-20 14:20",
        "likes": 1204,
        "comments": 203,
        "reposts": 388,
    },
    {
        "id": "mock-006",
        "text": "痛lo 日常化改造思路：把大柄换成小柄，或者加一件纯色罩衫遮住裙摆的印花，"
        "再配一双平底鞋，通勤地铁完全没问题。重点是不要戴满配饰，选一到两个重点就够了。",
        "author": "地铁里的lo娘",
        "created_at": "2026-02-18 08:35",
        "likes": 640,
        "comments": 87,
        "reposts": 96,
    },
    {
        "id": "mock-007",
        "text": "洗护踩坑记录：带亮片和烫金的柄图千万别机洗，我一条裙子洗完之后柄图整片开裂。"
        "正确做法是冷水手洗、中性洗涤剂、反面阴干；丝带和蝴蝶结最好拆下来单独洗。"
        "真丝和蕾丝建议直接送干洗。",
        "author": "洗裙子洗哭了",
        "created_at": "2026-02-15 16:55",
        "likes": 908,
        "comments": 141,
        "reposts": 267,
    },
    {
        "id": "mock-008",
        "text": "尺码问题统一回复：官方尺码表量的是平铺尺寸，一定要自己量好胸围、腰围、"
        "裙长三个数据再去对。如果卡在两个码之间，宁可选大的，找裁缝改小比改大容易得多。"
        "另外不同批次的同款可能有 1-2cm 误差，别只信别人说的「偏大偏小」。",
        "author": "尺码表复读机",
        "created_at": "2026-02-12 20:10",
        "likes": 733,
        "comments": 112,
        "reposts": 190,
    },
    {
        "id": "mock-009",
        "text": "二次元 IP 联动款真的越来越多了，这次联动的是我追了六年的番，"
        "柄图把主角的披风做成了裙摆的暗纹，细节拉满。缺点是预售期太长，等货等到失去热情。",
        "author": "等预售等到秃",
        "created_at": "2026-02-10 12:03",
        "likes": 455,
        "comments": 68,
        "reposts": 34,
    },
    {
        "id": "mock-010",
        "text": "手作发饰记录：用缎带做了一个 KC，成本不到 15 块，比买的现货便宜三分之二。"
        "教程有人要吗？材料和步骤我都拍了图。",
        "author": "手工部の小满",
        "created_at": "2026-02-08 09:26",
        "likes": 289,
        "comments": 52,
        "reposts": 17,
    },
    {
        "id": "mock-011",
        "text": "转发微博 @裙摆与猫：姐妹这条搭配太绝了，内搭的蕾丝领和柄图颜色呼应上了，"
        "求链接 https://t.cn/A6xyzQ 顺便问一下这个鞋是哪家的 #Lolita穿搭#",
        "author": "被种草的第三天",
        "created_at": "2026-02-06 18:40",
        "likes": 44,
        "comments": 9,
        "reposts": 2,
        "is_repost": True,
    },
    {
        "id": "mock-012",
        "text": "今天好热啊，气温 31 度直接穿短袖了，裙子是一点都不想穿，"
        "只想窝在家里吹空调吃西瓜。",
        "author": "怕热的阿凉",
        "created_at": "2026-02-05 15:12",
        "likes": 18,
        "comments": 3,
        "reposts": 0,
    },
    {
        "id": "mock-013",
        "text": "收盘了，今天大盘又是绿油油的一天，基金账户已经躺平不看。",
        "author": "韭菜的自我修养",
        "created_at": "2026-02-04 15:01",
        "likes": 7,
        "comments": 1,
        "reposts": 0,
    },
    {
        "id": "mock-014",
        "text": "整理了一份入坑清单，按顺序买最不容易踩坑：①一件百搭的白色内搭衬衫；"
        "②一条柄图和配色都喜欢的 JSK；③一双舒服的低跟玛丽珍；④一个能装的包（要放得下裙撑）。"
        "裙撑先买 A 字撑，不要一上来就买暴力撑，日常撑度 2-3 就够了。",
        "author": "理智剁手日记",
        "created_at": "2026-02-02 11:07",
        "likes": 1560,
        "comments": 240,
        "reposts": 512,
    },
]


def load_mock_posts() -> list[Post]:
    """返回 mock 语料（每次都返回新的对象，避免调用方互相污染）。"""
    posts: list[Post] = []
    for raw in MOCK_POSTS:
        data = dict(raw)
        data.setdefault("url", f"https://m.weibo.cn/detail/{data['id']}")
        data.setdefault("source", "mock")
        data["source"] = "mock"
        posts.append(Post.from_dict(data))
    return posts
