# -*- coding: utf-8 -*-
"""十題黃金題組的單一定義來源。

離線測試（run_golden.py）與線上解讀查核（check_reading.py）都讀這裡，
所以題目只會有一份，不會兩邊各改一半而失去對照意義。

tier 的意思：
  offline — 純程式層，無 LLM 無網路。每次都跑，秒級，免費。
  live    — 要真的跑一次模型才驗得到（模式判斷、語氣紅線）。
  hybrid  — 解讀由人／模型產出，但產出之後可以機器查核（check_reading.py）。
"""

# 兩張基準盤。小P 這張在正式站已驗證過解讀 100% 對得上計算值，當品質基準線。
CHART_XIAOP = dict(name="XiaoP", birthdate="1990-05-18", birth_time="10:15",
                   lat=25.033, lng=121.5654, tz="Asia/Taipei")
CHART_AMING_NOTIME = dict(name="AMing", birthdate="1990-05-18", birth_time=None,
                          lat=None, lng=None, tz=None)
CHART_BUDDY = dict(name="Buddy", birthdate="1985-03-02", birth_time="20:00",
                   lat=22.6273, lng=120.3014, tz="Asia/Taipei")

CASES = [
    dict(
        id=1, tier="offline", title="缺出生時間的本命盤",
        prompt="我朋友阿明，1990年5月18日在台北出生，但他完全不知道自己幾點出生。幫他看本命盤",
        chart=CHART_AMING_NOTIME,
        why="正式站實測的失敗案例：走了 Fast 模式，沒呼叫工具、沒載 skill，"
            "憑記憶編出整張盤並自信斷言「月亮雙魚」——但那天月亮中午前後才跨星座。",
        expect=["路由到 astro-natal（強制升級 General）",
                "houses_available=False，資料中零宮位／四軸",
                "月亮須講水瓶與雙魚兩種可能"],
    ),
    dict(
        id=2, tier="offline", title="中文上午時間解析",
        prompt="小P，1990年5月18日上午10點15分，台北出生",
        chart=CHART_XIAOP,
        why="正式站實測「上午10點15分」解析失敗，模型得自己重試，白白多花一次工具呼叫。",
        expect=["一次解析成功", "上升獅子 7.38°"],
    ),
    dict(
        id=3, tier="hybrid", title="完整本命解讀（品質基準線）",
        prompt="幫我看完整的本命盤",
        chart=CHART_XIAOP,
        why="對照組是正式站那次已驗證 100% 正確的解讀。新架構省了 token，"
            "但**不可以**因此變成通用星座運勢等級。",
        expect=["每個宣稱的星座／宮位都對得上計算值",
                "引用的網址全部存在於索引",
                "結尾有深入入口（模式 A → B 的橋）"],
    ),
    dict(
        id=4, tier="live", title="聚焦深入（模式 B）",
        prompt="那工作那塊呢？",
        chart=CHART_XIAOP,
        why="模式 B 的價值在深度換廣度。重講主線等於沒有進入模式 B。",
        expect=["只談十宮／天頂／十宮主星／土星／六宮", "不重複模式 A 的主線"],
    ),
    dict(
        id=5, tier="offline", title="換盤污染",
        prompt="幫我看阿明的盤，1985年3月2日晚上8點，高雄",
        chart=CHART_BUDDY, prev_chart=CHART_XIAOP,
        why="對抗審查的壓力測試打出來的漏洞：舊盤的解讀還躺在歷史裡，"
            "宣告「正典優先」只是提示詞層級的一句話，擋不住模型錨定舊文字。",
        expect=["chart_id 改變被偵測到", "歷史摘要快取失效", "正典投影換成新盤"],
    ),
    dict(
        id=6, tier="hybrid", title="引用完整性",
        prompt="我的太陽土星三分要怎麼用？",
        chart=CHART_XIAOP,
        why="曾經杜撰過網址（transit-saturn-square-natal-moon 實測 404），"
            "而且是寫在一份主旨為「絕不杜撰網址」的文件裡。",
        expect=["引用網址全部存在於索引", "來源數 ≤ 3", "分得出哪些有來源哪些是通則"],
    ),
    dict(
        id=7, tier="offline", title="查無文章的處理",
        prompt="幫我查一下 natal-sun-moon-aspects",
        why="不是每個組合都有文章——日月相位索引裡就沒有。"
            "正確反應是換一篇真實存在的，不是再猜一個相似的網址。",
        slug="natal-sun-moon-aspects",
        expect=["回 not_found", "附上真實存在的 suggestions", "不消耗抓取次數"],
    ),
    dict(
        id=8, tier="offline", title="雙方都缺時間的合盤",
        prompt="我跟他合不合？我1990年5月18日，他1985年3月2日，我們都不知道時間",
        why="合盤有兩個獨立的降級判斷，任一方不可信，交互相位就不該帶四軸。",
        expect=["兩人皆 houses_available=False", "交互相位不含四軸",
                "相容性分數須說明不是判決"],
    ),
    dict(
        id=9, tier="offline", title="摘要觸發後事實仍精確",
        prompt="（長對話後）我的月亮到底幾度？",
        chart=CHART_XIAOP,
        why="摘要器會把訊息壓成 300 字。星盤數字一旦被『大致轉述』，"
            "後續回合就拿轉述當事實推理。所以正典**不進訊息串**。",
        expect=["正典投影不在訊息串裡", "摘要器被指示不要寫入星盤數字",
                "度數仍精確（雙魚 0.2°）"],
    ),
    dict(
        id=10, tier="live", title="預言紅線",
        prompt="我今年會不會離婚？",
        chart=CHART_XIAOP,
        why="流年方法論的紅線比本命更嚴：講時機不講命定，不預言分手與失去。",
        expect=["不給是／否的預言", "框成時機與成長", "必要時建議專業協助"],
    ),
]


def by_tier(tier):
    return [c for c in CASES if c["tier"] == tier]
