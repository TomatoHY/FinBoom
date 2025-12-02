from typing import Optional, Any, Dict, Callable, Tuple, Union, List, Literal
import akshare as ak
import pandas as pd
import numpy as np
import time, re, requests, os, json
from io import StringIO
import unicodedata
from datetime import datetime, timedelta, date
import unittest
import traceback
import pickle
from functools import lru_cache
import tushare as ts
TUSHARE_API_KEY = 'efdf5a0a74f2b1163cf5979639fa3a779f37675151abf033219df8e2'
ts.set_token(TUSHARE_API_KEY)
CURRENCY_API_KEY = "pLV2InhpWP5q7PuoqsEQPEvck3fJZHBu"
_tushare_pro_api = None
_last_tushare_call_time = datetime.min
# 使用相对路径，相对于当前文件所在目录
LOCAL_PICKLE_FILE = os.path.join(os.path.dirname(__file__), 'local_data_archive.pkl') 

_stock_data_cache: Dict[str, Optional[pd.DataFrame]] = {}

STOCK_DATA_KEYS: Dict[str, Tuple[str, str]] = {
    'a_shares': ('a_name_code', '名称', '代码'),
    'hk_shares': ('hk_name_code', '中文名称', '代码'),
    'us_shares': ('us_name_code_market', '名称', '代码'),
}

_stock_data_cache: Dict[str, Optional[pd.DataFrame]] = {}

#-*- coding:utf-8 -*-    --------------Ashare 股票行情数据双核心版( https://github.com/mpquant/Ashare ) 
import json,requests;      import pandas as pd  #

#腾讯日线
def get_price_day_tx(code, end_date='', count=10, frequency='1d', adjust='qfq'):
    unit='week' if frequency in '1w' else 'month' if frequency in '1M' else 'day'
    if end_date:
        end_date=end_date.strftime('%Y-%m-%d') if isinstance(end_date, date) else end_date.split(' ')[0]
    end_date='' if end_date==datetime.now().strftime('%Y-%m-%d') else end_date
    adjust_param = 'qfq'
    if adjust == 'hfq':
        adjust_param = 'hfq'
    elif adjust == '':
        adjust_param = '' 
    URL=f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},{unit},,{end_date},{count},{adjust_param}'
    st= json.loads(requests.get(URL).content)
    if st.get('code') != 0 or 'data' not in st or code not in st['data']:
        print(f"  -> [Tencent] 失败: API 未返回 '{code}' 的有效数据。Msg: {st.get('msg')}")
        return None
    stk = st['data'][code]
    if not isinstance(stk, dict):
        print(f"  -> [Tencent] 失败: API 为 '{code}' 返回了非预期的格式 (非字典)。")
        return None
    ms = f'{adjust_param}{unit}' if adjust_param else unit
    buf=stk.get(ms)
    if buf is None:
        buf = stk.get(unit)
    if buf is None:
        print(f"  -> [Tencent] 失败: 在 API 响应中未找到 '{ms}' 或 '{unit}' 键。")
        return None
    df=pd.DataFrame(buf,columns=['time','open','close','high','low','volume'],dtype='float')
    df.time=pd.to_datetime(df.time);
    df.set_index(['time'], inplace=True);
    df.index.name='' 
    return df

#腾讯分钟线
def get_price_min_tx(code, end_date=None, count=10, frequency='1d'):
    ts=int(frequency[:-1]) if frequency[:-1].isdigit() else 1
    if end_date: end_date=end_date.strftime('%Y-%m-%d') if isinstance(end_date, date) else end_date.split(' ')[0]
    URL=f'http://ifzq.gtimg.cn/appstock/app/kline/mkline?param={code},m{ts},,{count}'
    try:
        st= json.loads(requests.get(URL).content)
        if 'data' not in st or code not in st['data'] or 'm'+str(ts) not in st['data'][code]:
            print(f"  -> [Tencent Min] 失败: API 未返回 'm{ts}' 数据 for {code}。")
            return None 
        buf = st['data'][code]['m'+str(ts)]
        if not buf:
            print(f"  -> [Tencent Min] 失败: API 返回了 'm{ts}' 的空列表 (code: {code} 可能不受支持)。")
            return None 
        df = pd.DataFrame(buf, columns=['time','open','close','high','low','volume','n1','n2'])
        df = df[['time','open','close','high','low','volume']]
        df[['open','close','high','low','volume']] = df[['open','close','high','low','volume']].astype('float')
        df.time=pd.to_datetime(df.time);  df.set_index(['time'], inplace=True);  df.index.name=''
        if not df.empty and 'qt' in st['data'][code] and code in st['data'][code]['qt']:
            df.iloc[-1, df.columns.get_loc('close')] = float(st['data'][code]['qt'][code][3])
        return df
    except Exception as e:
        print(f"  -> [Tencent Min] 失败: 处理 {code} 时发生意外错误: {e}")
        return None

#sina新浪全周期获取函数，分钟线 5m,15m,30m,60m  日线1d=240m   周线1w=1200m  1月=7200m
def get_price_sina(code, end_date='', count=10, frequency='60m'):
    frequency=frequency.replace('1d','240m').replace('1w','1200m').replace('1M','7200m');
    mcount=count
    ts=int(frequency[:-1]) if frequency[:-1].isdigit() else 1
    if (end_date!='') & (frequency in ['240m','1200m','7200m']): 
        end_date=pd.to_datetime(end_date) if not isinstance(end_date, date) else end_date
        unit=4 if frequency=='1200m' else 29 if frequency=='7200m' else 1
        count=count+(datetime.now()-end_date).days//unit
    URL=f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale={ts}&ma=5&datalen={count}' 
    dstr= json.loads(requests.get(URL).content);
    if not isinstance(dstr, list):
        print(f"  -> [Sina] 失败: API 未返回列表数据 (可能是不支持的 code: {code})。")
        return None 
    df= pd.DataFrame(dstr,columns=['day','open','high','low','close','volume'])
    df['open'] = df['open'].astype(float); df['high'] = df['high'].astype(float);
    df['low'] = df['low'].astype(float);  df['close'] = df['close'].astype(float);  df['volume'] = df['volume'].astype(float)
    df.day=pd.to_datetime(df.day)
    df.set_index(['day'], inplace=True)
    df.index.name=''
    if (end_date!='') & (frequency in ['240m','1200m','7200m']): 
        return df[df.index <= end_date][-mcount:]
    return df

def get_price(code, end_date='',count=10, frequency='1d', fields=[]):        
    xcode= code.replace('.XSHG','').replace('.XSHE','')                      #证券代码编码兼容处理 
    xcode='sh'+xcode if ('XSHG' in code)  else  'sz'+xcode  if ('XSHE' in code)  else code     
    if  frequency in ['1d','1w','1M']:   #1d日线  1w周线  1M月线
        try:    return get_price_sina( xcode, end_date=end_date,count=count,frequency=frequency)   #主力
        except: return get_price_day_tx(xcode,end_date=end_date,count=count,frequency=frequency)   #备用                    
    if  frequency in ['1m','5m','15m','30m','60m']:  #分钟线 ,1m只有腾讯接口  5分钟5m   60分钟60m
        if frequency in '1m': return get_price_min_tx(xcode,end_date=end_date,count=count,frequency=frequency)
        try:    return get_price_sina(  xcode,end_date=end_date,count=count,frequency=frequency)   #主力   
        except: return get_price_min_tx(xcode,end_date=end_date,count=count,frequency=frequency)   #备用

# Ashare 股票行情数据( https://github.com/mpquant/Ashare ) 
def _load_all_stock_data() -> bool:
    global _stock_data_cache
    if all(key in _stock_data_cache and _stock_data_cache[key] is not None for key in STOCK_DATA_KEYS):
        print("--- [Cache] 所有股票映射数据已加载。 ---")
        return True
    print(f"--- [Cache] 正在从 Pickle 文件 '{LOCAL_PICKLE_FILE}' 加载所有股票数据... ---")
    if not os.path.exists(LOCAL_PICKLE_FILE):
        print(f"--- [Cache] 错误: Pickle 文件 '{LOCAL_PICKLE_FILE}' 未找到。 ---")
        return False
    try:
        with open(LOCAL_PICKLE_FILE, 'rb') as f:
            data_archive = pickle.load(f)
        for key_alias, (archive_key, name_col, code_col) in STOCK_DATA_KEYS.items():
            if _stock_data_cache.get(key_alias) is None: 
                if archive_key in data_archive:
                    df = data_archive[archive_key]
                    if name_col not in df.columns:
                        print(f"--- [Cache] 严重错误: '{archive_key}' 中缺少 'STOCK_DATA_KEYS' 定义的名称列: '{name_col}'")
                        _stock_data_cache[key_alias] = None 
                        continue 
                    if code_col not in df.columns:
                        print(f"--- [Cache] 严重错误: '{archive_key}' 中缺少 'STOCK_DATA_KEYS' 定义的代码列: '{code_col}'")
                        _stock_data_cache[key_alias] = None 
                        continue 
                    df['normalized_name'] = df[name_col].astype(str).apply(normalize_stock_name)
                    _stock_data_cache[key_alias] = df
                else:
                    print(f"--- [Cache] 警告: Pickle 文件中缺少预期的键: '{archive_key}'。 ---")
                    _stock_data_cache[key_alias] = None
        return True
    except Exception as e:
        print(f"--- [Cache] 错误: 加载 Pickle 文件失败: {e} ---")
        return False

def normalize_stock_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^\w\u4e00-\u9fa5]', '', name)
    return name

def get_code_from_name(name: str, market: str = 'all') -> Optional[str]:
    """
    从股票名称查找股票代码。
    """
    global _stock_data_cache, STOCK_DATA_KEYS
    print(f"--- [LocalSearch] 开始为 '{name}' 在市场 '{market}' 中查找股票代码 ---")
    if not _load_all_stock_data():
        return f"--- [LocalSearch] 查找失败。无法加载本地数据归档。 ---"
    market_map = { 'a': 'a_shares', 'hk': 'hk_shares', 'us': 'us_shares' }
    sources_to_search = []
    if market in market_map:
        key = market_map[market]
        if key in STOCK_DATA_KEYS:
            sources_to_search.append(key)
    else: 
        sources_to_search = list(STOCK_DATA_KEYS.keys())
    normalized_input = normalize_stock_name(name)
    for cache_key in sources_to_search:
        df = _stock_data_cache.get(cache_key)
        if df is None or df.empty:
            continue
        _, name_col, code_col = STOCK_DATA_KEYS[cache_key] 
        normalized_df_names = df['normalized_name'] 
        exact_match_rows = df[normalized_df_names == normalized_input]
        if not exact_match_rows.empty:
            stock_row = exact_match_rows.iloc[0]
            raw_code = stock_row[code_col]
            stock_code = str(raw_code).split('.')[0] 
            print(f"*** [LocalSearch] 成功! 在 {cache_key} 中【精确匹配】到 '{stock_row[name_col]}' 的代码是: {stock_code} (原始值: {raw_code}) ***")
            return stock_code
        if len(normalized_input) > 1:
            contain_match_rows = df[normalized_df_names.str.contains(normalized_input, na=False)]
            if not contain_match_rows.empty:
                stock_row = contain_match_rows.iloc[0]
                raw_code = stock_row[code_col]
                stock_code = str(raw_code).split('.')[0]
                print(f"*** [LocalSearch] 成功! 在 {cache_key} 中【模糊匹配】到 '{stock_row[name_col]}' 的代码是: {stock_code} (原始值: {raw_code}) ***")
                return stock_code
    return f"--- [LocalSearch] 查找结束。在指定范围未能找到与 '{name}' 相关的代码。 ---"

_a_history_cache: Dict[tuple, pd.DataFrame] = {}
LATEST_KEYWORDS = ['最新', 'latest', 'newest', 'today', '今天', '当前', '实时']

def _fetch_a_history(symbol: str, adjust: str, start_date: str = "19700101", end_date: str = "22220101") -> Optional[pd.DataFrame]:
    global _a_history_cache
    cache_key = (symbol, adjust, start_date, end_date)
    if cache_key in _a_history_cache:
        print(f"--- [A股历史] 缓存命中: {symbol} ({adjust}) ---")
        return _a_history_cache[cache_key]
    df_hist, source = None, "未知"
    if symbol.startswith('bj'):
        print(f"--- [A股历史] 检测到北京 (BJ) 股票: {symbol}。正在使用 'ak.stock_bj_a_hist'... ---")
        try:
            symbol_no_prefix = symbol.replace("bj", "")
            df_raw = ak.stock_bj_a_hist(symbol=symbol_no_prefix, adjust=adjust)
            if df_raw is not None and not df_raw.empty:
                source = "Akshare/Beijing"
                df_hist = df_raw.copy()
        except Exception as e:
            print(f"--- [A股历史] 错误: 'ak.stock_bj_a_hist' 调用失败: {e}")
    else: 
        symbol_no_prefix = symbol.replace("sh", "").replace("sz", "")
        try:
            print(f"  -> 正在尝试从 [主源-Akshare/Eastmoney] 获取 '{symbol_no_prefix}' (adjust={adjust}) 的历史数据...")
            df_raw = ak.stock_zh_a_hist(
                symbol=symbol_no_prefix, period="daily",
                start_date=start_date.replace("-", ""), 
                end_date=end_date.replace("-", ""),   
                adjust=adjust
            )
            if df_raw is not None and not df_raw.empty:
                source = "Akshare/Eastmoney"
                df_hist = df_raw.copy()
        except Exception as e:
            print(f"  -> 从 [主源-Akshare/Eastmoney] 获取历史数据失败: {e}。将尝试备用源 1。")
        if df_hist is None and start_date == "19700101":
            print("--- [A股历史] 主源失败，正在回退到 [Ashare] (Sina/Tencent)... ---")
            try:
                print(f"  -> 正在尝试从 [备用源 1-Ashare/Sina] 获取 '{symbol}'...")
                df_raw = get_price_sina(symbol, end_date=end_date, count=99999, frequency='1d')
                if df_raw is not None and not df_raw.empty:
                    source = "Ashare/Sina"
                    df_raw.index.name = '日期' 
                    df_hist = df_raw.reset_index() 
                else:
                    raise ValueError("Sina 返回为空")
            except Exception as e_sina:
                print(f"  -> 从 [备用源 1-Ashare/Sina] 失败: {e_sina}。将尝试备用源 2。")
                try:
                    print(f"  -> 正在尝试从 [备用源 2-Ashare/Tencent] 获取 '{symbol}' (adjust={adjust})...")
                    df_raw = get_price_day_tx(symbol, end_date=end_date, count=99999, frequency='1d', adjust=adjust)
                    if df_raw is not None and not df_raw.empty:
                        source = "Ashare/Tencent"
                        df_raw.index.name = '日期'
                        df_hist = df_raw.reset_index()
                    else:
                        raise ValueError("Tencent 返回为空")
                except Exception as e_tx:
                    print(f"  -> 从 [备用源 2-Ashare/Tencent] 失败: {e_tx}。")
    if df_hist is None:
        print(f"--- [A股历史] 错误: 所有可用数据源均未能获取 '{symbol}' 的数据。 ---")
        return None
    print(f"--- [A股历史] 成功从 [{source}] 获取数据。正在进行标准化处理... ---")
    df_hist.rename(columns={
        'date': '日期', 
        'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低',
        'volume': '成交量', 'amount': '成交额'
    }, inplace=True, errors='ignore')
    if source in ["Akshare/Eastmoney", "Akshare/Beijing"]:
        if '成交量' in df_hist.columns:
            df_hist['成交量'] = df_hist['成交量'] * 100
    if '日期' not in df_hist.columns:
         print(f"--- [A股历史] 严重错误: 标准化后, '{source}' 的数据仍缺少 '日期' 列。")
         return None
    df_hist['日期'] = pd.to_datetime(df_hist['日期']).dt.strftime('%Y-%m-%d')
    df_hist.sort_values(by='日期', ascending=False, inplace=True)
    df_hist_final = df_hist.reset_index(drop=True)
    if source == "Akshare/Eastmoney":
        _a_history_cache[cache_key] = df_hist_final
    return df_hist_final

_a_spot_cache: Optional[pd.DataFrame] = None
_a_spot_cache_time: Optional[datetime] = None

def _fetch_a_spot_data() -> Optional[pd.DataFrame]:
    global _a_spot_cache, _a_spot_cache_time
    if _a_spot_cache is not None and (datetime.now() - _a_spot_cache_time).total_seconds() < 600:
        print("--- [A股实时] 缓存命中 (10分钟内)，跳过 API 调用。 ---")
        return _a_spot_cache
    print("--- [A股实时] 正在通过 ak.stock_zh_a_spot() 获取最新实时数据... ---")
    try:
        time.sleep(30)
        df = ak.stock_zh_a_spot() 
        if df is None or df.empty:
            print("--- [A股实时] 错误: API未返回任何实时数据。 ---")
            return None
        df['normalized_name'] = df['名称'].astype(str).apply(normalize_stock_name)
        _a_spot_cache = df
        _a_spot_cache_time = datetime.now()
        print(f"--- [A股实时] 成功获取 {len(df)} 条数据并已缓存。 ---")
        return df
    except Exception as e:
        print(f"--- [A股实时] 错误: 从 ak.stock_zh_a_spot 获取数据失败: {e} ---")
        return None

def _get_tushare_pro_api():
    """ 辅助函数: 初始化并返回 Tushare Pro API 实例。"""
    global _tushare_pro_api
    
    if _tushare_pro_api:
        return _tushare_pro_api
    try:
        # ！！！ 在这里填入你的 Token ！！！
        USER_TUSHARE_TOKEN = "TUSHARE_API_KEY" 
        ts.set_token(USER_TUSHARE_TOKEN) 
        _tushare_pro_api = ts.pro_api()
        _tushare_pro_api.trade_cal(exchange='SSE', start_date='20200101', end_date='20200101')
        print("--- [Tushare] Tushare Pro API 初始化并连接成功。 ---")
    except Exception as e:
        print(f"--- [Tushare] Tushare Pro API 初始化失败 (Token 是否正确?): {e} ---")
        _tushare_pro_api = None
        return None
    return _tushare_pro_api

def _apply_tushare_rate_limit():
    """ 辅助函数: 强制执行 Tushare 速率限制 (2 次/分钟)。"""
    global _last_tushare_call_time
    seconds_since_last_call = (datetime.now() - _last_tushare_call_time).total_seconds()
    if seconds_since_last_call < 30.1: 
        wait_time = 30.1 - seconds_since_last_call
        print(f"--- [Tushare Rate Limit] 2 calls/min. Pausing for {wait_time:.1f} seconds... ---")
        time.sleep(wait_time)
    _last_tushare_call_time = datetime.now()


def _fetch_a_realtime_hybrid(
    resolved_symbol: str, 
    resolved_ts_code: str,
    target_col: str 
) -> Tuple[Optional[pd.Series], str]:
    print(f"--- [A股实时] 正在尝试 [P1 主源 Ashare] get_price(code={resolved_symbol}, freq=1d)... ---")
    try:
        df_ashare = get_price(code=resolved_symbol, frequency='1d', count=1) 
        if df_ashare is not None and not df_ashare.empty:
            realtime_row = df_ashare.iloc[-1].copy() 
            realtime_row['日期'] = realtime_row.name.strftime('%Y-%m-%d %H:%M:%S')
            realtime_row['开盘'] = realtime_row['open']
            realtime_row['收盘'] = realtime_row['close']
            realtime_row['最高'] = realtime_row['high']
            realtime_row['最低'] = realtime_row['low']
            realtime_row['成交量'] = realtime_row['volume']
            if target_col in realtime_row:
                return realtime_row, "Ashare (Sina/Tencent)"
            else:
                print(f"  -> [Ashare] 成功, 但缺少 '{target_col}' 列。正在回退...")
        else:
            print(f"  -> [Ashare] 失败: API 返回为空。")
    except Exception as e:
        print(f"  -> [Ashare] 失败: {e}")
    print(f"--- [A股实时] [P1 Ashare] 失败或缺少数据。正在回退到 [P2 备用源 Akshare]... ---")
    try:
        df_akshare_spot = _fetch_a_spot_data() 
        if df_akshare_spot is not None:
            match = df_akshare_spot[df_akshare_spot['代码'] == resolved_symbol]
            if not match.empty:
                realtime_row = match.iloc[0].copy()
                # [标准化] Akshare -> 中文
                realtime_row['日期'] = datetime.now().strftime('%Y-%m-%d')
                realtime_row['开盘'] = realtime_row['今开']
                realtime_row['收盘'] = realtime_row['最新价']
                realtime_row['最高'] = realtime_row['最高']
                realtime_row['最低'] = realtime_row['最低']
                realtime_row['成交量'] = realtime_row['成交量']
                realtime_row['成交额'] = realtime_row['成交额'] # <-- Akshare 有 '成交额'
                if target_col in realtime_row:
                    return realtime_row, "Akshare (Spot)"
                else:
                    print(f"  -> [Akshare Spot] 成功, 但缺少 '{target_col}' 列。正在回退...")
    except Exception as e:
        print(f"  -> [Akshare Spot] 失败: {e}")
    print(f"--- [A股实时] [P2 Akshare] 失败或缺少数据。正在回退到 [P3 备用源 Tushare]... ---")
    pro = _get_tushare_pro_api()
    if not pro:
        print("  -> [Tushare] 失败: Tushare API 未初始化。")
        return None, "Tushare (Failed)"
    try:
        _apply_tushare_rate_limit()
        df_tushare = pro.rt_k(ts_code=resolved_ts_code)
        if df_tushare is not None and not df_tushare.empty:
            realtime_row = df_tushare.iloc[0].copy()
            realtime_row['日期'] = realtime_row['trade_time']
            realtime_row['开盘'] = realtime_row['open']
            realtime_row['收盘'] = realtime_row['close']
            realtime_row['最高'] = realtime_row['high']
            realtime_row['最低'] = realtime_row['low']
            realtime_row['成交量'] = realtime_row['vol']
            realtime_row['成交额'] = realtime_row['amount'] 
            if target_col in realtime_row:
                return realtime_row, "Tushare (rt_k)"
            else:
                print(f"  -> [Tushare rt_k] 成功, 但缺少 '{target_col}' 列。")
    except Exception as e:
        print(f"  -> [Tushare rt_k] 失败: {e}")
    print(f"--- [A股实时] 错误: 所有 3 个实时数据源均失败, 或都缺少 '{target_col}'。 ---")
    return None, "All Failed"


def _fetch_a_history_hybrid(
    resolved_symbol: str, 
    resolved_ts_code: str, 
    resolved_ak_code: str,
    adjust: str,
    start_date: str,
    end_date: str
) -> Tuple[Optional[pd.DataFrame], str]:
    df_hist = None
    source = "Unknown"
    if adjust != 'hfq' and not resolved_symbol.startswith('bj'):
        print(f"--- [A股历史] 正在尝试 [P1 主源 Tushare] pro.daily(ts_code={resolved_ts_code})... ---")
        pro = _get_tushare_pro_api()
        if not pro:
            print("  -> [Tushare] 失败: Tushare API 未初始化。")
        else:
            try:
                _apply_tushare_rate_limit()
                start_yyyy = start_date.replace("-", "")
                end_yyyy = end_date.replace("-", "")
                df_hist = pro.daily(ts_code=resolved_ts_code, start_date=start_yyyy, end_date=end_yyyy)
                if df_hist is not None and not df_hist.empty:
                    source = "Tushare (pro.daily)"
                    df_hist['日期'] = pd.to_datetime(df_hist['trade_date'], format='%Y%m%d')
                    df_hist['开盘'] = df_hist['open']
                    df_hist['收盘'] = df_hist['close']
                    df_hist['最高'] = df_hist['high']
                    df_hist['最低'] = df_hist['low']
                    df_hist['昨收'] = df_hist['pre_close']
                    df_hist['涨跌幅'] = df_hist['pct_chg']
                    df_hist['成交量'] = df_hist['vol'] * 100   # (手 -> 股)
                    df_hist['成交额'] = df_hist['amount'] * 1000 # (千元 -> 元)
                else:
                    print(f"  -> [Tushare Daily] 失败: pro.daily 未返回数据。")
                    df_hist = None
            except Exception as e:
                print(f"  -> [Tushare Daily] 失败: pro.daily 调用失败: {e}")
                df_hist = None
    else:
        print(f"--- [A股历史] [P1 Tushare] 跳过 (不支持 hfq 或 bj 股)。")
    if df_hist is None:
        print(f"--- [A股历史] [P1 Tushare] 失败或跳过。正在回退到 [P2 备用源 四核]... ---")
        try:
            df_hist = _fetch_a_history(
                symbol=resolved_symbol, 
                adjust=adjust,
                start_date=start_date,
                end_date=end_date
            ) 
            source = "Akshare/Ashare (Four-Core)"
        except Exception as e:
            print(f"  -> [四核 _fetch_a_history] 失败: {e}")
            df_hist = None
    if df_hist is None: 
        print(f"--- [A股历史] 错误: 所有数据源 (Tushare, Akshare/Ashare) 均失败。 ---")
        return None, "All Failed"
    return df_hist, source

def get_a_stock_daily_price(
    column_label: str,
    adjust: str,
    query_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name: Optional[str] = None,
    code: Optional[str] = None,
    original_query: Optional[str] = None
) -> Dict[str, Any]: 
    """
    获取A股实时或历史行情数据。
    """
    COLUMN_MAPPING = {
        'open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘',
        'volume': '成交量', 'amount': '成交额',
        '开盘': '开盘', '最高': '最高', '最低': '最低',
        '收盘': '收盘', '最新价': '收盘', 
        '成交量': '成交量', '成交额': '成交额',
    }
    def _fail(error_msg: str) -> Dict[str, Any]:
        return { "result": error_msg, "min_value": None,
                "requested_item": {"value": None, "error": error_msg},
                "date": datetime.now().strftime('%Y-%m-%d') }
    if adjust not in ['', 'qfq', 'hfq']: return _fail(f"错误: 'adjust' 参数 '{adjust}' 无效。")
    if not code and not name: return _fail("错误: 必须提供股票代码 (code) 或股票名称 (name)。")
    if not query_date and not start_date: return _fail("错误: 必须提供 `query_date` 或 `start_date`。")
    identifier = name if name else code 
    resolved_symbol = None  
    resolved_ts_code = None 
    resolved_ak_code = None 
    try:
        if code:
            resolved_symbol = str(code)
        elif name:
            print(f"--- [A股] 'code' 未提供, 正在使用 'name' ({name}) 从 [本地缓存] 查找代码... ---")
            found_code = get_code_from_name(name=name, market='a') 
            if not found_code or "--- [LocalSearch] 查找" in str(found_code):
                return _fail(f"错误: 无法通过名称 '{name}' 从 [本地缓存] 找到对应的股票代码。{found_code}")
            resolved_symbol = str(found_code)
        if not resolved_symbol:
            return _fail(f"错误: 无法解析 '{identifier}' 为有效的股票代码。")
        if resolved_symbol.startswith('sh'):
            resolved_ak_code = resolved_symbol[2:]
            resolved_ts_code = resolved_ak_code + ".SH"
        elif resolved_symbol.startswith('sz'):
            resolved_ak_code = resolved_symbol[2:]
            resolved_ts_code = resolved_ak_code + ".SZ"
        elif resolved_symbol.startswith('bj'):
            resolved_ak_code = resolved_symbol[2:]
            resolved_ts_code = resolved_ak_code + ".BJ"
        else:
            return _fail(f"错误: 解析的代码 '{resolved_symbol}' 缺少 'sh', 'sz' 或 'bj' 前缀。")
    except Exception as e:
        return _fail(f"在为 '{identifier}' 解析代码时失败: {e}")
    is_latest_query = (query_date and query_date.lower().strip() in LATEST_KEYWORDS) or \
                    (original_query and any(k in original_query.lower() for k in LATEST_KEYWORDS))
    if is_latest_query:
        print(f"--- [A股实时] 执行 [{identifier}] 的实时数据查询 (Ashare > Akshare > Tushare) ---")
        clean_column_label = column_label.lower().strip()
        hist_col = COLUMN_MAPPING.get(clean_column_label) # [新] 简化的映射
        if not hist_col:
            return _fail(f"错误: 列名 '{clean_column_label}' 在 COLUMN_MAPPING 中未定义。")
        realtime_row, source = _fetch_a_realtime_hybrid(
            resolved_symbol=resolved_symbol,
            resolved_ts_code=resolved_ts_code,
            target_col=hist_col 
        )
        if realtime_row is None:
            return _fail(f"错误: 所有实时源 (Ashare, Akshare, Tushare) 均未能获取 '{identifier}' 的数据。")
        try:
            clean_column_label = column_label.lower().strip()
            hist_col = COLUMN_MAPPING.get(clean_column_label) # [新] 简化的映射
            if not hist_col:
                return _fail(f"错误: 列名 '{clean_column_label}' 在 COLUMN_MAPPING 中未定义。")
            if hist_col not in realtime_row:
                return _fail(f"错误: 实时数据 (源: {source}) 中缺少必需列 '{hist_col}' (映射自 '{clean_column_label}')。")
            value = realtime_row[hist_col]
            query_date_str = realtime_row['日期']
            return { "result": float(value), "stock_identifier": resolved_symbol, "market": "A-Share",
                    "date": query_date_str, "requested_item": {"name": clean_column_label, "value": float(value)} }
        except Exception as e:
            return _fail(f"处理 '{identifier}' (源: {source}) 的实时数据时发生未知错误: {e}")
    print(f"--- [A股历史] 执行 [{identifier}] 的历史数据查询 (Tushare > 四核备) ---")
    df_hist, source = _fetch_a_history_hybrid(
        resolved_symbol=resolved_symbol,
        resolved_ts_code=resolved_ts_code,
        resolved_ak_code=resolved_ak_code,
        adjust=adjust,
        start_date=start_date if start_date else "19700101",
        end_date=end_date or query_date or datetime.now().strftime('%Y-%m-%d')
    )
    if df_hist is None: 
        return _fail(f"错误: 所有历史源 (Tushare, Akshare/Ashare) 均无法获取 '{identifier}' 的历史数据。")
    try:
        clean_column_label = column_label.lower().strip()
        hist_col = COLUMN_MAPPING.get(clean_column_label)
        if not hist_col or hist_col not in df_hist.columns:
            valid_cols = [col for col in ['开盘','收盘','最高','最低','成交量','成交额'] if col in df_hist.columns]
            return _fail(f"错误: A股历史数据(源: {source})中列名 '{clean_column_label}' (映射: {hist_col}) 无效。可用列: {valid_cols}")
        if start_date: # 范围查询
            effective_end_date = end_date or query_date 
            if effective_end_date is None:
                effective_end_date = datetime.now().strftime('%Y-%m-%d')
            df_hist['日期'] = pd.to_datetime(df_hist['日期'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(effective_end_date)
            range_df = df_hist[(df_hist['日期'] >= start_dt) & (df_hist['日期'] <= end_dt)].copy()
            if range_df.empty: 
                return _fail(f"错误: 在指定日期范围 {start_date} 到 {effective_end_date} 内未找到任何数据。")
            range_df[hist_col] = pd.to_numeric(range_df[hist_col], errors='coerce')
            min_row = range_df.loc[range_df[hist_col].idxmin()]
            min_val = float(min_row[hist_col])
            return {
                "result": min_val, "query_type": "range_minimum", "stock_identifier": resolved_symbol, 
                "min_value": min_val, "date_of_min_value": min_row['日期'].strftime('%Y-%m-%d'),
                "requested_item": {"name": clean_column_label, "value": min_val}
            }
        else: # 单点查询
            df_hist['日期'] = pd.to_datetime(df_hist['日期'])
            target_dt = pd.to_datetime(query_date)
            row_found = pd.DataFrame()
            print(f"--- 未找到精确日期 '{query_date}'，正在回退查找最近的有效交易日... ---")
            for i in range(7): # 7 天回溯
                current_target_dt = target_dt - timedelta(days=i)
                row_found = df_hist[df_hist['日期'] == current_target_dt]
                if not row_found.empty:
                    break
            if row_found.empty:
                min_d, max_d = df_hist['日期'].min(), df_hist['日期'].max()
                return _fail(f"错误: 未找到日期 '{query_date}' 或任何更早的数据。可用数据范围: {min_d} 到 {max_d}。")
            value = float(row_found.iloc[0][hist_col])
            actual_date = row_found.iloc[0]['日期'].strftime('%Y-%m-%d')
            return {
                "result": value, "query_type": "single_date", "stock_identifier": resolved_symbol,
                "date": actual_date, "requested_item": {"name": clean_column_label, "value": value}
            }
    except Exception as e:
        return _fail(f"处理 '{identifier}' (源: {source}) 的历史数据时发生未知错误: {e}")

_us_spot_cache: Optional[pd.DataFrame] = None
_us_spot_cache_time: Optional[datetime] = None
US_SPOT_CACHE_EXPIRY_SECONDS = 600 

def _fetch_us_spot_data() -> Optional[pd.DataFrame]:
    global _us_spot_cache, _us_spot_cache_time
    if _us_spot_cache is not None and (datetime.now() - _us_spot_cache_time).total_seconds() < US_SPOT_CACHE_EXPIRY_SECONDS:
        print("--- [美股实时] 缓存命中 (10分钟内)，跳过 API 调用。 ---")
        return _us_spot_cache
    print("--- [美股实时] 正在通过 ak.stock_us_spot() 获取最新实时数据 (全量)... ---")
    try:
        time.sleep(30) 
        df = ak.stock_us_spot() 
        if df is None or df.empty:
            print("--- [美股实时] 错误: API未返回任何实时数据。 ---")
            return None
        df['normalized_name'] = (df['name'].astype(str).apply(normalize_stock_name) + " " + df['cname'].astype(str).apply(normalize_stock_name))
        df['symbol'] = df['symbol'].astype(str).str.upper() # 确保 symbol 大写
        _us_spot_cache = df
        _us_spot_cache_time = datetime.now()
        print(f"--- [美股实时] 成功获取 {len(df)} 条数据并已缓存。 ---")
        return df
    except Exception as e:
        print(f"--- [美股实时] 错误: 从 ak.stock_us_spot 获取数据失败: {e} ---")
        return None

def get_us_stock_daily_price(
    column_label: str,
    adjust: str,
    query_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name: Optional[str] = None,
    code: Optional[str] = None,
    original_query: Optional[str] = None
) -> Dict[str, Any]:
    COLUMN_MAPPING = {
        'open':           {'spot_us': 'open', 'hist': '开盘'},
        'high':           {'spot_us': 'high', 'hist': '最高'},
        'low':            {'spot_us': 'low', 'hist': '最低'},
        'close':          {'spot_us': 'price', 'hist': '收盘'},
        'latest_price':   {'spot_us': 'price', 'hist': '收盘'},
        'volume':         {'spot_us': 'volume', 'hist': '成交量'},
        'amount':         {'spot_us': 'mktcap', 'hist': '成交额'}, 
        'market_cap':     {'spot_us': 'mktcap', 'hist': '成交额'}, 
        'change_percent': {'spot_us': 'chg_percent', 'hist': '涨跌幅'},
        'pe_ratio':       {'spot_us': 'pe', 'hist': '市盈率(TTM)'},
    }
    def _fail(error_msg: str) -> Dict[str, Any]:
        return {
            "result": error_msg,
            "min_value": None,
            "requested_item": {"value": None, "error": error_msg}
        }
    if adjust not in ['', 'qfq', 'hfq']: return _fail(f"错误: 'adjust' 参数 '{adjust}' 无效。")
    if not code and not name: return _fail("错误: 必须提供股票代码 (code) 或股票名称 (name)。")
    if not query_date and not start_date: return _fail("错误: 必须提供 `query_date` 或 `start_date`。")
    identifier = name if name else code 
    is_latest_query = (query_date and query_date.lower().strip() in LATEST_KEYWORDS) or \
                    (original_query and any(k in original_query.lower() for k in LATEST_KEYWORDS))
    if is_latest_query:
        print(f"--- [美股实时] 执行 [{identifier}] 的实时数据查询 (来源: API + Cache) ---")
        df_realtime = _fetch_us_spot_data()
        if df_realtime is None:
            return _fail(f"错误: 无法从 API 获取实时数据快照 (ak.stock_us_spot)。")
        realtime_row_match = pd.DataFrame() 
        search_ticker = ""
        if code:
            search_ticker = code.upper()
            realtime_row_match = df_realtime[df_realtime['symbol'] == search_ticker]
        elif name:
            normalized_input = normalize_stock_name(name)
            realtime_row_match = df_realtime[df_realtime['normalized_name'].str.contains(normalized_input, na=False)]
            if not realtime_row_match.empty: 
                search_ticker = realtime_row_match.iloc[0]['symbol']
        if realtime_row_match.empty:
            return _fail(f"错误: 在实时 API 数据中未找到 '{identifier}'。")
        realtime_row = realtime_row_match.iloc[0]
        clean_column_label = column_label.lower().strip()
        spot_col_name = COLUMN_MAPPING.get(clean_column_label, {}).get('spot_us')
        if not spot_col_name:
            return _fail(f"错误: 列名 '{clean_column_label}' 在美股 COLUMN_MAPPING 中没有定义 'spot_us' 键。")
        if spot_col_name not in realtime_row:
            return _fail(f"错误: 实时数据中缺少必需列 '{spot_col_name}' (映射自 '{clean_column_label}')。")
        value = realtime_row[spot_col_name]
        if pd.isna(value): 
            return _fail(f"错误: 实时数据中 '{clean_column_label}' 的值不可用。")
        query_date_str = datetime.now().strftime('%Y-%m-%d')
        return {
            "result": float(value),
            "requested_item": {"name": clean_column_label, "value": float(value)},
            "min_value": float(value), 
            "query_type": "single_date", "stock_identifier": search_ticker,
            "date": query_date_str,
        }
    print(f"--- [美股历史] 执行 [{identifier}] 的历史数据查询 (来源: API接口) ---")
    if not code:
        return _fail(f"错误: 历史查询必须提供 'code' (股票代码, e.g., 'MSFT')。")
    search_ticker = code.upper()
    possible_symbols = [
        f"105.{search_ticker}", # 纳斯达克 (e.g., MSFT, AAPL, NVDA)
        f"106.{search_ticker}", # 纽约 (e.g., TTE, JPM)
        f"107.{search_ticker}"  # 美交所
    ]
    df_hist = None
    symbol_for_hist = ""
    for symbol in possible_symbols:
        try:
            print(f"--- [美股历史] 正在尝试使用 Symbol '{symbol}' 调用 ak.stock_us_hist()... ---")
            df_hist_attempt = ak.stock_us_hist(symbol=symbol, adjust=adjust, start_date="19700101", end_date="20991231") 
            if df_hist_attempt is not None and not df_hist_attempt.empty:
                df_hist = df_hist_attempt
                symbol_for_hist = symbol
                print(f"--- [美股历史] 成功: Symbol '{symbol}' 返回了数据。")
                break 
            else:
                print(f"--- [美股历史] 失败: Symbol '{symbol}' 未返回数据。")
        except Exception as e:
            print(f"--- [美股历史] 失败: Symbol '{symbol}' 调用失败: {e}")
        print("   -> (暂停 2 秒...)")
        time.sleep(30) 
    if df_hist is None:
        return _fail(f"错误: 无法使用任何猜测的 symbol (e.g., 105.{search_ticker}, 106.{search_ticker}) 获取 '{search_ticker}' 的历史数据。")
    try:
        if '日期' not in df_hist.columns:
            return _fail(f"错误: 获取到的历史数据缺少 '日期' 列。")
        df_hist['日期'] = pd.to_datetime(df_hist['日期']).dt.strftime('%Y-%m-%d')
        df_hist.sort_values(by='日期', ascending=False, inplace=True)
        df_hist.reset_index(drop=True, inplace=True)
        clean_column_label = column_label.lower().strip()
        hist_col = COLUMN_MAPPING.get(clean_column_label, {}).get('hist')
        if not hist_col or hist_col not in df_hist.columns:
            valid_cols = [k for k, v in COLUMN_MAPPING.items() if v.get('hist') in df_hist.columns]
            return _fail(f"错误: 美股历史数据中列名 '{clean_column_label}' 无效。可用列: {valid_cols}")
        if start_date: # 范围查询
            effective_end_date = end_date or query_date 
            if effective_end_date is None:
                effective_end_date = datetime.now().strftime('%Y-%m-%d')
            range_df = df_hist[(df_hist['日期'] >= start_date) & (df_hist['日期'] <= effective_end_date)].copy()
            if range_df.empty:
                return _fail(f"错误: 在美股 {start_date} 到 {effective_end_date} 内未找到任何数据。")
            min_value_row = range_df.loc[range_df[hist_col].idxmin()]
            min_val = float(min_value_row[hist_col])
            return {
                "result": min_val, 
                "requested_item": {"name": clean_column_label, "value": min_val},
                "query_type": "range_minimum", 
                "stock_identifier": search_ticker,
                "min_value": min_val, 
                "date_of_min_value": min_value_row['日期']
            }
        else: # 单点查询
            row_found = df_hist[df_hist['日期'] == query_date]
            if row_found.empty:
                print(f"--- [美股历史] 未找到精确日期 '{query_date}'，正在回退... ---")
                row_found = df_hist[df_hist['日期'] <= query_date].head(1)
            if row_found.empty:
                return _fail(f"错误: 未找到美股日期 '{query_date}' 或更早的数据。")
            value = float(row_found.iloc[0][hist_col])
            return {
                "result": value, 
                "requested_item": {"name": clean_column_label, "value": value},
                "min_value": value, 
                "query_type": "single_date", "stock_identifier": search_ticker,
                "date": row_found.iloc[0]['日期'],
            }
    except Exception as e:
        return _fail(f"处理 '{search_ticker}' 的美股历史数据时发生未知错误: {e}")

_hk_spot_cache: Optional[pd.DataFrame] = None
_hk_spot_cache_time: Optional[datetime] = None
HK_SPOT_CACHE_EXPIRY_SECONDS = 600 
MAX_API_RETRIES = 3 
RETRY_SLEEP_SECONDS = 120 

def _fetch_hk_spot_data() -> Optional[pd.DataFrame]:
    global _hk_spot_cache, _hk_spot_cache_time
    if _hk_spot_cache is not None and (datetime.now() - _hk_spot_cache_time).total_seconds() < HK_SPOT_CACHE_EXPIRY_SECONDS:
        print("--- [港股实时] 缓存命中 (10分钟内)，跳过 API 调用。 ---")
        return _hk_spot_cache
    print(f"--- [港股实时] 正在通过 ak.stock_hk_spot() 获取最新实时数据 (全量)... ---")
    for attempt in range(MAX_API_RETRIES):
        try:
            print(f"  -> 尝试 {attempt + 1}/{MAX_API_RETRIES}...")
            time.sleep(10 * (attempt + 1)) 
            df = ak.stock_hk_spot()
            if df is not None and not df.empty:
                print(f"--- [港股实时] 成功获取数据 (尝试 {attempt + 1})。 ---")
                df['normalized_name'] = (
                    df['中文名称'].astype(str).apply(normalize_stock_name) + " " +
                    df['英文名称'].astype(str).apply(normalize_stock_name)
                )
                df['代码'] = df['代码'].astype(str).str.strip().str.zfill(5)
                _hk_spot_cache = df
                _hk_spot_cache_time = datetime.now()
                return df
            print(f"  -> 尝试 {attempt + 1} 失败: API返回空数据。")
        except Exception as e:
            print(f"  -> ❌ 尝试 {attempt + 1} 失败: {type(e).__name__} - {e}")
        if attempt < MAX_API_RETRIES - 1:
            print(f"  -> {RETRY_SLEEP_SECONDS} 秒后重试...")
            time.sleep(RETRY_SLEEP_SECONDS)
    print(f"--- [港股实时] ❌ 错误: 经过 {MAX_API_RETRIES} 次尝试后仍无法获取数据。 ---")
    return None

def _fetch_hk_history(symbol: str, adjust: str) -> Optional[pd.DataFrame]:
    """
    获取港股历史数据，并尝试使用两个数据源。
    """
    df_hist, source = None, "未知"
    try:
        print(f"  -> 正在尝试从 [主源-新浪] 获取 '{symbol}' 的历史数据...")
        time.sleep(30) 
        df_raw = ak.stock_hk_daily(symbol=symbol, adjust=adjust)
        if df_raw is not None and not df_raw.empty:
            source = "新浪"
            df_hist = df_raw.rename(columns={
                'date': '日期', 'open': '开盘', 'high': '最高', 'low': '最低',
                'close': '收盘', 'volume': '成交量'
            })
    except Exception as e:
        print(f"  -> 从新浪获取历史数据失败: {e}。将尝试备用源。")
    
    if df_hist is None:
        print(f"  -> (暂停 120 秒后尝试备用源...)")
        time.sleep(120) 
        try:
            print(f"  -> 正在尝试从 [备用源-东财] 获取 '{symbol}' 的历史数据...")
            df_raw = ak.stock_hk_hist(symbol=symbol, adjust=adjust)
            if df_raw is not None and not df_raw.empty:
                source = "东方财富"
                df_hist = df_raw
        except Exception as e:
            print(f"  -> 从备用源东方财富获取历史数据也失败: {e}")
            
    if df_hist is None: return None
    
    print(f"--- 成功从 [{source}] 获取数据。正在进行标准化处理... ---")
    df_hist['日期'] = pd.to_datetime(df_hist['日期']).dt.strftime('%Y-%m-%d')
    df_hist.sort_values(by='日期', ascending=False, inplace=True)
    return df_hist.reset_index(drop=True)

_hk_history_cache_tushare: Dict[str, pd.DataFrame] = {}
_hk_history_cache_times_tushare: Dict[str, datetime] = {}
_hk_history_cache_akshare: Dict[Tuple, Optional[pd.DataFrame]] = {}
HK_STOCK_CACHE_EXPIRY_SECONDS = 3600 # 缓存 1 小时

def _get_tushare_pro_api():
    """ 辅助函数: 初始化并返回 Tushare Pro API 实例。"""
    global _tushare_pro_api
    if _tushare_pro_api:
        return _tushare_pro_api
    try:
        USER_TUSHARE_TOKEN = "efdf5a0a74f2b1163cf5979639fa3a779f37675151abf033219df8e2" 
        ts.set_token(USER_TUSHARE_TOKEN) 
        _tushare_pro_api = ts.pro_api()
        _tushare_pro_api.trade_cal(exchange='SSE', start_date='20200101', end_date='20200101')
        print("--- [Tushare] Tushare Pro API 初始化并连接成功。 ---")
    except Exception as e:
        print(f"--- [Tushare] Tushare Pro API 初始化失败 (Token 是否正确?): {e} ---")
        _tushare_pro_api = None
        return None
    return _tushare_pro_api

def _apply_tushare_rate_limit():
    """ 辅助函数: 强制执行 2 次/分钟 (1 次/30秒) 的速率限制。"""
    global _last_tushare_call_time
    seconds_since_last_call = (datetime.now() - _last_tushare_call_time).total_seconds()
    if seconds_since_last_call < 30.1: 
        wait_time = 30.1 - seconds_since_last_call
        print(f"--- [Tushare Rate Limit] 2 calls/min. Pausing for {wait_time:.1f} seconds... ---")
        time.sleep(wait_time)
    _last_tushare_call_time = datetime.now() 

def _fetch_hk_history_from_tushare(ts_code: str) -> Optional[pd.DataFrame]:
    """
    辅助函数: 从 Tushare 获取单只港股的完整历史数据。
    """
    pro = _get_tushare_pro_api()
    if not pro:
        return None
    try:
        print(f"--- [Tushare API] 正在调用 pro.hk_daily(ts_code='{ts_code}')... ---")
        _apply_tushare_rate_limit() 
        df = pro.hk_daily(
            ts_code=ts_code,
            start_date="19900101", 
            end_date=datetime.now().strftime('%Y%m%d')
        )
        if df is None or df.empty:
            print(f"--- [Tushare API] 错误: pro.hk_daily 未返回 '{ts_code}' 的任何数据。 ---")
            return None
        df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"--- [Tushare API] 错误: pro.hk_daily 调用失败: {e} ---")
        return None


def get_hk_stock_daily_price(
    column_label: str,
    adjust: str = '',
    query_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name: Optional[str] = None,
    code: Optional[str] = None,
    original_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取港股实时或历史行情数据。
    """
    COLUMN_MAPPING = {
        'open': {'spot_hk': '今开', 'hist': '开盘'},
        'high': {'spot_hk': '最高', 'hist': '最高'},
        'low': {'spot_hk': '最低', 'hist': '最低'},
        'close': {'spot_hk': '最新价', 'hist': '收盘'},
        'latest_price': {'spot_hk': '最新价', 'hist': '收盘'},
        'volume': {'spot_hk': '成交量', 'hist': '成交量'},
        'amount': {'spot_hk': '成交额', 'hist': '成交额'},
        'change_percent': {'spot_hk': '涨跌幅', 'hist': '涨跌幅'}, 
        'pct_change': {'spot_hk': '涨跌幅', 'hist': '涨跌幅'},
        '开盘':   {'spot_hk': '今开', 'hist': '开盘'},
        '最高':   {'spot_hk': '最高', 'hist': '最高'},
        '最低':   {'spot_hk': '最低', 'hist': '最低'},
        '收盘':   {'spot_hk': '最新价', 'hist': '收盘'},
        '最新价': {'spot_hk': '最新价', 'hist': '收盘'},
        '成交量': {'spot_hk': '成交量', 'hist': '成交量'},
        '成交额': {'spot_hk': '成交额', 'hist': '成交额'},
        '涨跌幅': {'spot_hk': '涨跌幅', 'hist': '涨跌幅'},
    }
    def _fail(error_msg: str) -> Dict[str, Any]:
        return {
            "result": error_msg,
            "min_value": None, 
            "requested_item": {"value": None, "error": error_msg}, 
            "date": datetime.now().strftime('%Y-%m-%d') 
        }
    if adjust not in ['', 'qfq', 'hfq']: return _fail(f"错误: 'adjust' 参数 '{adjust}' 无效。")
    if not code and not name: return _fail("错误: 必须提供股票代码 (code) 或股票名称 (name)。")
    if not query_date and not start_date: return _fail("错误: 必须提供 `query_date` 或 `start_date`。")
    identifier = name if name else code 
    is_latest_query = (query_date and query_date.lower().strip() in LATEST_KEYWORDS) or \
                    (original_query and any(k in original_query.lower() for k in LATEST_KEYWORDS))
    if is_latest_query:
        print(f"--- [港股实时] 执行 [{identifier}] 的实时数据查询 (来源: Akshare) ---")
        df_realtime = _fetch_hk_spot_data()
        if df_realtime is None:
            return _fail(f"错误: 无法从 API 获取实时数据快照 (ak.stock_hk_spot)。")
        realtime_row_match = pd.DataFrame()
        search_code = ""
        if code:
            search_code = str(code).zfill(5)
            realtime_row_match = df_realtime[df_realtime['代码'] == search_code]
        elif name: 
            normalized_input = normalize_stock_name(name)
            exact_match = df_realtime[df_realtime['normalized_name'] == normalized_input]
            if not exact_match.empty:
                realtime_row_match = exact_match
                search_code = realtime_row_match.iloc[0]['代码']
            else:
                contain_match = df_realtime[df_realtime['normalized_name'].str.contains(normalized_input, na=False)]
                if not contain_match.empty:
                    realtime_row_match = contain_match.head(1)
                    search_code = realtime_row_match.iloc[0]['代码']
        if realtime_row_match.empty:
            return _fail(f"错误: 在实时 API 数据中未找到 '{identifier}'。")
        realtime_row = realtime_row_match.iloc[0]
        display_name = realtime_row['中文名称']
        clean_column_label = column_label.lower().strip()
        spot_col_name = COLUMN_MAPPING.get(clean_column_label, {}).get('spot_hk')
        if not spot_col_name:
            return _fail(f"错误: 列名 '{clean_column_label}' 在 COLUMN_MAPPING 中没有定义 'spot_hk' 键。")
        if spot_col_name not in realtime_row:
            fallback_col = '最新' if spot_col_name == '最新价' else None
            if fallback_col and fallback_col in realtime_row:
                spot_col_name = fallback_col
            else:
                return _fail(f"错误: 实时数据中缺少必需列 '{spot_col_name}' (映射自 '{clean_column_label}')。")
        value = realtime_row[spot_col_name]
        if pd.isna(value): 
            return _fail(f"错误: 实时数据中 '{clean_column_label}' (API列: {spot_col_name}) 的值不可用。")
        query_date_str = datetime.now().strftime('%Y-%m-%d')
        return {
            "result": float(value), 
            "query_type": "single_date", 
            "stock_identifier": search_code,
            "date": query_date_str,
            "requested_item": {"name": clean_column_label, "value": float(value)}
        }
    print(f"--- [港股历史] 执行 [{identifier}] 的历史数据查询 (Tushare 主 / Akshare 备) ---")
    symbol_akshare = None 
    symbol_tushare = None 
    display_name = identifier 
    if code:
        symbol_akshare = str(code).zfill(5)
        symbol_tushare = f"{symbol_akshare}.HK"
        display_name = symbol_akshare
    elif name:
        print(f"--- [港股历史] 'name' ({name}) 已提供, 正在 [Akshare 实时缓存] 查找代码... ---")
        df_realtime_cache = _fetch_hk_spot_data()
        if df_realtime_cache is None:
            return _fail(f"错误: 无法获取实时数据快照，无法通过名称找到代码。")
        normalized_input = normalize_stock_name(name)
        exact_match = df_realtime_cache[df_realtime_cache['normalized_name'] == normalized_input]
        stock_row = None
        if not exact_match.empty:
            stock_row = exact_match.iloc[0]
        else:
            contain_match = df_realtime_cache[df_realtime_cache['normalized_name'].str.contains(normalized_input, na=False)]
            stock_row = contain_match.head(1).iloc[0] if not contain_match.empty else None
        if stock_row is None:
            return _fail(f"错误: 无法在 API 实时数据中通过名称 '{name}' 找到代码。")
        symbol_akshare = str(stock_row['代码']).zfill(5)
        symbol_tushare = f"{symbol_akshare}.HK"
        display_name = stock_row['中文名称']
        print(f"--- [港股历史] 成功在API缓存中找到代码: {symbol_akshare} / {symbol_tushare} ---")
    if not symbol_tushare or not symbol_akshare:
        return _fail(f"错误: 历史查询未能确定股票代码。")
    if adjust:
        print(f"--- [港股历史] 警告: Tushare 'pro.hk_daily' (主源) 不支持 'adjust' (复权)。将首先尝试获取未复权数据。")
        print(f"--- [港股历史] Akshare 'stock_hk_daily' (备用源) 支持复权。")
    df_hist = None
    source = "Unknown"
    global _hk_history_cache_tushare, _hk_history_cache_times_tushare
    df_hist = _hk_history_cache_tushare.get(symbol_tushare)
    cache_time = _hk_history_cache_times_tushare.get(symbol_tushare)
    if df_hist is None or (datetime.now() - cache_time).total_seconds() > HK_STOCK_CACHE_EXPIRY_SECONDS:
        print(f"--- [HK Cache] Tushare 缓存未命中/过期。正在调用 [主源 Tushare]... ---")
        df_hist = _fetch_hk_history_from_tushare(ts_code=symbol_tushare)
        if df_hist is not None and not df_hist.empty:
            _hk_history_cache_tushare[symbol_tushare] = df_hist
            _hk_history_cache_times_tushare[symbol_tushare] = datetime.now()
            print(f"--- [HK Tushare] 成功获取并缓存 {len(df_hist)} 条数据。")
            source = "Tushare"
        else:
            df_hist = None 
            print(f"--- [HK Tushare] [主源 Tushare] 失败。")
    else:
        print(f"--- [HK Cache] Tushare 缓存命中 ('{symbol_tushare}')。 ---")
        source = "Tushare"
    if df_hist is None:
        print(f"--- [HK Akshare] 正在回退到 [备用源 Akshare]... ---")
        global _hk_history_cache_akshare
        cache_key_akshare = (symbol_akshare, adjust)
        df_hist = _hk_history_cache_akshare.get(cache_key_akshare)
        if df_hist is None:
            df_hist = _fetch_hk_history(symbol=symbol_akshare, adjust=adjust) # <-- 你的旧 Akshare 调用
            if df_hist is None: 
                return _fail(f"错误: [主源 Tushare] 和 [备用源 Akshare] 均无法获取 '{display_name}' 的历史数据。")
            _hk_history_cache_akshare[cache_key_akshare] = df_hist
            print("--- [HK Akshare] 成功从备用源获取数据。")
        else:
            print("--- [HK Akshare] 备用源缓存命中。")
        source = "Akshare"
    if source == "Tushare":
        print("--- [HK] 正在标准化 Tushare 列名以匹配 Akshare 逻辑... ---")
        df_hist.rename(columns={
            'date': '日期',      
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'pre_close': '昨收',
            'change': '涨跌额',
            'pct_chg': '涨跌幅',
            'vol': '成交量',
            'amount': '成交额'
        }, inplace=True)
    try:
        clean_column_label = column_label.lower().strip()
        hist_col = COLUMN_MAPPING.get(clean_column_label, {}).get('hist')
        if not hist_col or hist_col not in df_hist.columns:
            if source == "Tushare" and clean_column_label == 'prev_close':
                hist_col = '昨收'
            if not hist_col or hist_col not in df_hist.columns:
                valid_cols = [v.get('hist') for v in COLUMN_MAPPING.values() if v.get('hist') in df_hist.columns]
                return _fail(f"错误: 历史数据(源: {source})中列名 '{clean_column_label}' (映射: {hist_col}) 无效。可用列: {valid_cols}")
        if start_date: # 范围查询
            effective_end_date = end_date or query_date 
            if effective_end_date is None:
                effective_end_date = datetime.now().strftime('%Y-%m-%d')
            df_hist['日期'] = pd.to_datetime(df_hist['日期'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(effective_end_date)
            range_df = df_hist[(df_hist['日期'] >= start_dt) & (df_hist['日期'] <= end_dt)]
            if range_df.empty: 
                return _fail(f"错误: 在指定日期范围 {start_date} 到 {effective_end_date} 内未找到任何数据。")
            range_df[hist_col] = pd.to_numeric(range_df[hist_col], errors='coerce')
            min_row = range_df.loc[range_df[hist_col].idxmin()]
            min_val = float(min_row[hist_col])
            return {
                "result": min_val, 
                "query_type": "range_minimum", 
                "stock_identifier": symbol_akshare, 
                "min_value": min_val, 
                "date_of_min_value": min_row['日期'].strftime('%Y-%m-%d'),
                "requested_item": {"name": clean_column_label, "value": min_val}
            }
        else: # 单点查询
            df_hist['日期'] = pd.to_datetime(df_hist['日期'])
            target_dt = pd.to_datetime(query_date)
            row_found = pd.DataFrame()
            print(f"--- 正在查找精确日期 '{query_date}' (带7天回溯)... ---")
            for i in range(7):
                current_target_dt = target_dt - timedelta(days=i)
                row_found = df_hist[df_hist['日期'] == current_target_dt]
                if not row_found.empty:
                    break
            if row_found.empty:
                min_d, max_d = df_hist['日期'].min(), df_hist['日期'].max()
                return _fail(f"错误: 未找到日期 '{query_date}' 或任何更早的数据。可用数据范围: {min_d} 到 {max_d}。")
            value = float(row_found.iloc[0][hist_col])
            actual_date = row_found.iloc[0]['日期'].strftime('%Y-%m-%d')
            return {
                "result": value, 
                "query_type": "single_date", 
                "stock_identifier": symbol_akshare,
                "date": actual_date,
                "requested_item": {"name": clean_column_label, "value": value}
            }
    except Exception as e:
        return _fail(f"处理 '{display_name}' (源: {source}) 的历史数据时发生未知错误: {e}")
    
def calculate_a_stock_change_pct_in_period(
    start_date: str,
    end_date: str,
    adjust: str,
    name: Optional[str] = None,
    code: Optional[str] = None
) -> str:
    """
    计算A股某只股票在【指定时间段内】的累计涨跌幅（百分比）。
    """
    if adjust not in ['', 'qfq', 'hfq']:
        return f"错误: 'adjust' 参数 '{adjust}' 无效。有效选项: '', 'qfq', 'hfq'。"
    if not code and not name:
        return "错误：必须提供股票代码 (code) 或股票名称 (name)。"
    try:
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_date_ak = start_date_dt.strftime('%Y%m%d')
        end_date_ak = end_date_dt.strftime('%Y%m%d')
    except ValueError:
        return f"错误: 日期格式无效。请确保 start_date 和 end_date 均为 'YYYY-MM-DD' 格式。"
    if start_date_dt >= end_date_dt:
        return f"错误: 开始日期 {start_date} 必须早于结束日期 {end_date}。"
    symbol = code if code else get_code_from_name(name, market='a')
    if not symbol:
        return f"错误：未能通过名称 '{name}' 或代码 '{code}' 找到对应的A股代码。"
    try:
        hist_df = ak.stock_zh_a_hist_tx(
            symbol=symbol,
            adjust=adjust,
            start_date=start_date_ak,
            end_date=end_date_ak
        )
        if hist_df is None or hist_df.empty:
            return f"错误: 在指定时间段内未能获取到代码'{symbol}'的任何交易数据。请检查代码或日期范围。"
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_df.sort_values(by='date', inplace=True)
    except Exception as e:
        return f"下载代码'{symbol}'的历史数据时发生错误: {e}"
    try:
        first_day_data = hist_df.iloc[0]
        last_day_data = hist_df.iloc[-1]
        start_price = first_day_data['close']
        end_price = last_day_data['close']
        actual_start_date = first_day_data['date'].strftime('%Y-%m-%d')
        actual_end_date = last_day_data['date'].strftime('%Y-%m-%d')
        if start_price == 0:
            return f"错误：起始交易日 {actual_start_date} 的收盘价为0，无法计算涨跌幅。"
        change_pct = ((end_price - start_price) / start_price) * 100
        result_json = {
            "stock_name": name,
            "stock_code": symbol,
            "query_period_start": start_date,
            "query_period_end": end_date,
            "actual_trading_day_start": actual_start_date,
            "actual_trading_day_end": actual_end_date,
            "start_price": f"{start_price:.2f}",
            "end_price": f"{end_price:.2f}",
            "change_percentage": f"{change_pct:+.2f}%"
        }
        return result_json
    except (KeyError, IndexError) as e:
        return f"错误：返回的数据格式不正确，无法提取收盘价或首末日期。({e})"
    except Exception as e:
        return f"计算涨跌幅时发生错误: {e}"
    
def calculate_futures_spread(
    symbol1: str,
    symbol2: str,
    start_date: str,
    end_date: str
) -> str:
    """
    计算两种期货合约在指定时间段内的【每日价差 (symbol2 - symbol1)】，
    并找出价差的最大值、最小值及其对应的日期。
    """
    try:
        print(f"--- 正在获取大连商品交易所(DCE)从 {start_date} 到 {end_date} 的全部日线数据... ---")
        market_df = ak.get_futures_daily(start_date=start_date, end_date=end_date, market="DCE")
        if market_df.empty:
            return f"错误: 未能获取到在 {start_date} 到 {end_date} 期间大连商品交易所的任何数据。"
        df1 = market_df[market_df['symbol'] == symbol1].copy()
        if df1.empty:
            return f"错误: 在获取到的市场数据中，未能找到合约 '{symbol1}' 的记录。"
        df2 = market_df[market_df['symbol'] == symbol2].copy()
        if df2.empty:
            return f"错误: 在获取到的市场数据中，未能找到合约 '{symbol2}' 的记录。"
        df1['date'] = pd.to_datetime(df1['date'])
        df2['date'] = pd.to_datetime(df2['date'])
        df1.set_index('date', inplace=True)
        df2.set_index('date', inplace=True)
        df1_close = df1[['close']].rename(columns={'close': f'close_{symbol1}'})
        df2_close = df2[['close']].rename(columns={'close': f'close_{symbol2}'})
        merged_df = pd.merge(df1_close, df2_close, left_index=True, right_index=True, how='inner')
        if merged_df.empty:
            return "错误: 两种合约在指定时间段内没有共同的交易日，无法计算价差。"
        merged_df['spread'] = merged_df[f'close_{symbol2}'] - merged_df[f'close_{symbol1}']
        max_spread_row = merged_df.loc[merged_df['spread'].idxmax()]
        max_spread_value = max_spread_row['spread']
        max_spread_date = max_spread_row.name.strftime('%Y-%m-%d')
        min_spread_row = merged_df.loc[merged_df['spread'].idxmin()]
        min_spread_value = min_spread_row['spread']
        min_spread_date = min_spread_row.name.strftime('%Y-%m-%d')
        start_date_display = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
        end_date_display = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
        result_json = {
            "analysis_type": "futures_spread",
            "symbol_base": symbol1,
            "symbol_target": symbol2,
            "calculation_formula": f"{symbol2} - {symbol1}",
            "query_period_start": start_date_display,
            "query_period_end": end_date_display,
            "max_spread": {
                "value": f"{max_spread_value:.2f}",
                "date": max_spread_date
            },
            "min_spread": {
                "value": f"{min_spread_value:.2f}",
                "date": min_spread_date
            }
        }
        return result_json
    except Exception as e:
        import traceback
        return f"计算期货价差时发生未知错误: {e}\n{traceback.format_exc()}"

def _fail_minute_spread(error_msg: str) -> Dict[str, Any]:
    """
    [辅助函数] 确保 calculate_futures_minute_spread 
    """
    print(f"[Debug] [calculate_futures_minute_spread] 失败: {error_msg}")
    print(f"[Debug] [calculate_futures_minute_spread] 正在返回哨兵值 0.0 以防止 solve() 崩溃。")
    return {
        'max_spread': {
            'value': 0.0, 
            'date': "1970-01-01 00:00:00", 
            'error': error_msg
        },
        'min_spread': { 
            'value': 0.0,
            'date': "1970-01-01 00:00:00",
            'error': error_msg
        }
    }

def calculate_futures_minute_spread(
    symbol1: str,
    symbol2: str,
    period: str = "1" 
) -> Dict[str, Any]: 
    """
    计算两种期货合约在【当日分钟线】上的价差 (symbol2 - symbol1)，
    并找出价差的最大值、最小值及其对应的分钟。
    """
    try:
        s1_upper = symbol1.upper()
        s2_upper = symbol2.upper()
        print(f"--- [分钟查询] 正在获取 {s1_upper} (周期: {period}min) 的分时数据... ---")
        df1 = ak.futures_zh_minute_sina(symbol=s1_upper, period=period)
        if df1 is None or df1.empty:
            return _fail_minute_spread(f"错误: 未能获取到 {s1_upper} 的分时数据。")
        time.sleep(30) 
        print(f"--- [分钟查询] 正在获取 {s2_upper} (周期: {period}min) 的分时数据... ---")
        df2 = ak.futures_zh_minute_sina(symbol=s2_upper, period=period)
        if df2 is None or df2.empty:
            return _fail_minute_spread(f"错误: 未能获取到 {s2_upper} 的分时数据。")
        df1['datetime'] = pd.to_datetime(df1['datetime'])
        df2['datetime'] = pd.to_datetime(df2['datetime'])
        df_merged = pd.merge(
            df1[['datetime', 'close']], 
            df2[['datetime', 'close']], 
            on='datetime', 
            suffixes=(f'_{s1_upper}', f'_{s2_upper}')
        )
        if df_merged.empty:
            return _fail_minute_spread("错误: 两种合约在当日分时数据中没有共同的时间戳，无法计算价差。")
        df_merged['spread'] = df_merged[f'close_{s2_upper}'] - df_merged[f'close_{s1_upper}']
        max_spread_row = df_merged.loc[df_merged['spread'].idxmax()]
        max_spread_value = max_spread_row['spread']
        max_spread_datetime = max_spread_row['datetime'].strftime('%Y-%m-%d %H:%M:%S')
        min_spread_row = df_merged.loc[df_merged['spread'].idxmin()]
        min_spread_value = min_spread_row['spread']
        min_spread_datetime = min_spread_row['datetime'].strftime('%Y-%m-%d %H:%M:%S')
        result_json = {
            "analysis_type": "futures_minute_spread",
            "symbol_base": symbol1,
            "symbol_target": symbol2,
            "calculation_formula": f"{symbol2} - {symbol1}",
            "max_spread": {
                "value": float(max_spread_value),
                "date": max_spread_datetime # [注意] 键是 'date', 但值是 'datetime'
            },
            "min_spread": {
                "value": float(min_spread_value),
                "date": min_spread_datetime
            }
        }
        return result_json
    except Exception as e:
        return _fail_minute_spread(f"计算期货(分钟)价差时发生未知异常: {e}\n{traceback.format_exc()}")
    
LATEST_KEYWORDS = ['最新', 'latest', 'newest', 'today', '今天', '当前', '实时']
_index_data_cache: Dict[str, Optional[pd.DataFrame]] = {}
_us_index_history_cache: Dict[str, Optional[pd.DataFrame]] = {}
INDEX_API_MAP: Dict[str, callable] = {
    'hk_index': lambda: ak.stock_hk_index_spot_sina(),
    'a_index': lambda: ak.stock_zh_index_spot_sina(),
    'global_index': lambda: ak.index_global_spot_em(), 
    'us_index': lambda: pd.DataFrame() 
}
_COLUMN_MAP_AK_TO_STD = {
    'date': 'date', 'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low',
    'volume': 'volume', 'turnover': 'turnover', '日期': 'date', '开盘': 'open',
    '今开': 'open', '收盘': 'close', '最新价': 'close', 'latest': 'close', '最高': 'high',
    '高': 'high', '最低': 'low', '低': 'low', '成交量': 'volume', '成交额': 'turnover',
    'amount': 'turnover',
}
def normalize_name(s: str) -> str:
    """标准化指数名称 (移除空格)"""
    if not isinstance(s, str): return ""
    return s.lower().replace(" ", "")
ALIAS_TO_STANDARD_NAME_MAP = {
        normalize_name("S&P 500"): "标普500",
        normalize_name("spx"): "标普500",
        normalize_name("dow jones"): "道琼斯工业平均指数",
        normalize_name("dow"): "道琼斯工业平均指数",
        normalize_name("djia"): "道琼斯工业平均指数",
        normalize_name("nasdaq"): "纳斯达克综合指数",
        normalize_name("ixic"): "纳斯达克综合指数",
        normalize_name("hsi"): "恒生指数",
        normalize_name("台湾加权指数"): "台湾加权",
        normalize_name("长三角指数"): "长三角",
        normalize_name("雅加达综合股价指数"): "印尼雅加达综合",
        normalize_name("jakarta stock exchange composite"): "印尼雅加达综合",
        normalize_name("idx composite"): "印尼雅加达综合",
        normalize_name("jkse"): "印尼雅加达综合",
        normalize_name('Nikkei 225'): "N225",
        normalize_name("S&P/ASX 200"): "AS51",
        normalize_name("德国DAX指数"): "德国DAX30",
        normalize_name("DAX Index"): '德国DAX30',
        normalize_name("SMI"): 'SSMI',
        normalize_name("Swiss Market Index"): "SSMI"
    }

_index_spot_caches: Dict[str, Optional[pd.DataFrame]] = {}
_index_spot_cache_times: Dict[str, Optional[datetime]] = {}
INDEX_SPOT_CACHE_EXPIRY_SECONDS = 3600 # 缓存过期时间设置为 1h

def _fetch_index_spot_api(market_key: str, api_func: callable) -> Optional[pd.DataFrame]:
    if market_key in _index_spot_caches and (datetime.now() - _index_spot_cache_times.get(market_key, datetime.min)).total_seconds() < INDEX_SPOT_CACHE_EXPIRY_SECONDS:
        print(f"--- [实时指数] 缓存命中 ({market_key})，跳过 API 调用。 ---")
        return _index_spot_caches[market_key]
    print(f"--- [实时指数] 正在通过 API ({market_key}) 获取最新数据... ---")
    try:
        time.sleep(30) 
        df = api_func() 
        if df is None or df.empty:
            print(f"--- [实时指数] 错误: {market_key} API未返回任何数据。 ---")
            return None
        if market_key == 'us_index':
            if 'symbol' in df.columns:
                df['代码'] = df['symbol'].astype(str).str.upper() # 将 symbol 映射到 '代码'
        _index_spot_caches[market_key] = df
        _index_spot_cache_times[market_key] = datetime.now()
        print(f"--- [实时指数] 成功获取 {len(df)} 条 {market_key} 数据并已缓存。 ---")
        return df
    except Exception as e:
        print(f"--- [实时指数] 错误: {market_key} API 调用失败: {e} ---")
        return None

_index_data_cache: Dict[str, Optional[pd.DataFrame]] = {}
_us_index_history_cache: Dict[str, Optional[pd.DataFrame]] = {} # 历史缓存不变
INDEX_DATA_KEYS: Dict[str, Tuple[str, str, str]] = {
    # Key: (Pickle Archive Key, Name Column, Code Column)
    'a_index': ('a_index_data', '名称', '代码'),
    'hk_index': ('hk_index_data', '名称', '代码'),
    'global_index': ('global_index_data', '名称', '代码'),
    'us_index': ('us_index_data', '名称', '代码'), 
}

def _load_all_index_data() -> bool:
    global _index_data_cache
    # 修复后的代码
    # 检查："是否 INDEX_DATA_KEYS 中的所有键都存在于缓存中，并且它们的值都不是 None？"
    if all(key in _index_data_cache and _index_data_cache[key] is not None for key in INDEX_DATA_KEYS):
        print("--- [Cache] 所有指数映射数据已加载。 ---")
        return True
    print(f"--- [Cache] 正在从 Pickle 文件 '{LOCAL_PICKLE_FILE}' 加载所有指数数据... ---")
    if not os.path.exists(LOCAL_PICKLE_FILE):
        print(f"--- [Cache] 错误: Pickle 文件 '{LOCAL_PICKLE_FILE}' 未找到。 ---")
        return False
    try:
        with open(LOCAL_PICKLE_FILE, 'rb') as f:
            data_archive = pickle.load(f)
        success = True
        for key_alias, (archive_key, name_col, code_col) in INDEX_DATA_KEYS.items():
            if archive_key in data_archive:
                df = data_archive[archive_key]
                df['normalized_name'] = df[name_col].astype(str).str.strip().apply(normalize_name)
                df[code_col] = df[code_col].astype(str).str.strip().str.lower()
                _index_data_cache[key_alias] = df
            else:
                print(f"--- [Cache] 警告: Pickle 文件中缺少预期的键: '{archive_key}'。 ---")
                _index_data_cache[key_alias] = None
                success = False 
        return success
    except Exception as e:
        print(f"--- [Cache] 错误: 加载 Pickle 文件失败: {e} ---")
        return False
    
def find_index_code_and_market(identifier: str, market_hint: Optional[str] = None) -> Optional[Tuple[str, str, str]]:
    global _index_data_cache
    print(f"--- [辅助函数] 正在为 '{identifier}' 查找信息 (市场提示: {market_hint})... ---")
    if not _load_all_index_data():
        print("--- [辅助函数] 查找失败。无法加载指数数据归档。 ---")
        return None
    US_INDEX_MAP = {
        ".IXIC": ("纳斯达克综合指数", ".IXIC", "IXIC"),
        "IXIC": ("纳斯达克综合指数", ".IXIC", "IXIC"),
        ".DJI": ("道琼斯工业平均指数", ".DJI", "DJI"),
        "DJI": ("道琼斯工业平均指数", ".DJI", "DJI"),
        ".INX": ("标普500", ".INX", "INX"),
        "INX": ("标普500", ".INX", "INX"),
        ".NDX": ("纳斯达克100", ".NDX", "NDX"),
        "NDX": ("纳斯达克100", ".NDX", "NDX"),
    }
    search_key_upper = identifier.upper().replace(".", "")
    if search_key_upper in US_INDEX_MAP:
        name, code, _ = US_INDEX_MAP[search_key_upper]
        print(f"--- [辅助函数] 识别为US指数 '{name}' (代码: {code})")
        return code, name, 'us_index'
    search_list = []
    if market_hint:
        hint_key = f"{market_hint.lower()}_index"
        sorted_keys = sorted(INDEX_DATA_KEYS.keys(), key=lambda x: x != hint_key)
        search_list = [(key, INDEX_DATA_KEYS[key]) for key in sorted_keys]
    else:
        search_list = list(INDEX_DATA_KEYS.items())
    normalized_input = normalize_name(identifier)
    standard_name = ALIAS_TO_STANDARD_NAME_MAP.get(normalized_input)
    search_target_norm = normalize_name(standard_name) if standard_name else normalized_input
    for market_key, (archive_key, name_col, code_col) in search_list:
        df = _index_data_cache.get(market_key) 
        if df is None or df.empty: continue
        code_search_input = identifier.lower()
        df_codes_as_string = df[code_col] 
        code_match = df[df_codes_as_string == code_search_input]
        if not code_match.empty:
            row = code_match.iloc[0]
            print(f"--- [辅助函数] 通过代码 '{identifier}' 在 '{market_key}' 中找到 '{row[name_col]}'")
            return str(row[code_col]), str(row[name_col]), market_key
        normalized_df_names = df['normalized_name'] # 已在加载时创建
        exact_match = df[normalized_df_names == search_target_norm]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            print(f"--- [辅助函数] 通过名称 '{identifier}' 在 '{market_key}' 中找到 '{row[name_col]}'")
            return str(row[code_col]), str(row[name_col]), market_key
    for market_key, (archive_key, name_col, code_col) in search_list:
        df = _index_data_cache.get(market_key)
        if df is None or df.empty: continue
        normalized_df_names = df['normalized_name'] # 已在加载时创建
        fuzzy_target = normalize_name(identifier)
        if len(fuzzy_target) < 3: continue 
        contain_match = df[normalized_df_names.str.contains(fuzzy_target, na=False)]
        if not contain_match.empty:
            row = contain_match.iloc[0]
            print(f"--- [辅助函数] 通过名称 (模糊匹配) '{identifier}' 在 '{market_key}' 中找到 '{row[name_col]}'")
            return str(row[code_col]), str(row[name_col]), market_key
    print(f"--- [辅助函数] 未能找到 '{identifier}' 对应的任何信息。 ---")
    return None

def get_index_realtime(
    identifier: str,
    column_label: str,
    market: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取指数的【实时】行情数据。
    """
    print(f"--- [本地实时查询] 正在为 '{identifier}' 查找 '{column_label}'... ---")
    entity_info = find_index_code_and_market(identifier=identifier, market_hint=market)
    if not entity_info:
        return {"value": f"错误: 实体链接失败，未能找到与 '{identifier}' 相关的信息。"}
    target_code, original_name, market_key = entity_info
    clean_column_label = column_label.lower().strip()
    value = None
    date_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    found_col_name = "N/A"
    try:
        if market_key == 'us_index':
            us_api_func = lambda: ak.index_us_stock_sina(symbol=target_code)
            print(f"--- [US实时] 正在通过 ak.index_us_stock_sina(symbol='{target_code}') 获取数据... ---")
            time.sleep(30)
            df_raw = us_api_func()
            if df_raw is None or df_raw.empty:
                raise ValueError(f"无法从 {target_code} API 获取实时数据。")
            df_raw.rename(columns={'close': '最新价', 'preclose': '昨收价'}, inplace=True)
            stock_row = df_raw.iloc[-1]
            date_time_str = stock_row.get('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            if clean_column_label == 'pct_change':
                latest_close = stock_row.get('最新价')
                prev_close = stock_row.get('昨收价')
                if pd.isna(latest_close) or pd.isna(prev_close) or prev_close == 0:
                    raise ValueError(f"无法计算涨跌幅，缺少最新价/昨收价。")
                value = ((latest_close - prev_close) / prev_close) * 100.0 # Guaranteed float
                found_col_name = '计算涨跌幅'
            else:
                COLUMN_MAP_US_SEARCH = {'open': 'open', 'high': 'high', 'low': 'low', 'close': '最新价', 'prev_close': '昨收价'}
                search_col = COLUMN_MAP_US_SEARCH.get(clean_column_label)
                if not search_col or search_col not in stock_row:
                    raise ValueError(f"US指数不支持查询 '{column_label}'，或缺少数据。")
                value = stock_row[search_col]
                found_col_name = search_col
        else:
            df_realtime = _fetch_index_spot_api(market_key, INDEX_API_MAP[market_key])
            if df_realtime is None:
                raise ValueError(f"无法从 API 获取 {market_key} 市场的实时数据快照。")
            code_col_name = '代码' if market_key in ['a_index', 'hk_index'] else 'symbol' 
            if code_col_name not in df_realtime.columns:
                code_col_name = 'code' 
                if code_col_name not in df_realtime.columns:
                    raise ValueError(f"实时数据 API 结构异常，缺少代码列。")
            match = df_realtime[df_realtime[code_col_name].astype(str).str.lower() == target_code.lower()]
            if match.empty:
                raise ValueError(f"在实时 API 数据中未能找到代码为 '{target_code}' 的记录。")
            stock_row = match.iloc[0]
            COLUMN_MAP_STD_TO_LOCAL = {
                'open': ['今开', '开盘', '开盘价'], 'high': ['最高', '最高价'],
                'low': ['最低', '最低价'], 'close': ['最新价'],
                'prev_close': ['昨收', '昨收价'], 'change': ['涨跌额'],
                'pct_change': ['涨跌幅'], 'volume': ['成交量'],
                'amount': ['成交额'], 'amplitude': ['振幅'],
                'time': ['时间', '最新行情时间', '日期'] 
            }
            local_col_names = COLUMN_MAP_STD_TO_LOCAL.get(clean_column_label)
            if not local_col_names:
                raise ValueError(f"查询的列标签 '{column_label}' 不受支持。")
            for col in local_col_names:
                if col in stock_row and pd.notna(stock_row[col]):
                    found_col_name = col
                    value = stock_row[found_col_name]
                    break
            if found_col_name == "N/A":
                raise ValueError(f"在记录中缺少或 '{column_label}' 的有效数据。")
            date_time_str = stock_row.get(COLUMN_MAP_STD_TO_LOCAL['time'][0], datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        return {"value": f"错误: {market_key} 实时查询失败。 {error_msg}"}
    if clean_column_label in ['pct_change', 'change', 'close', 'open', 'high', 'low']:
        if not isinstance(value, (float, int)):
            value = str(value)
        if isinstance(value, str):
            try:
                s_value = value.strip().rstrip('%').replace('+', '').replace(',', '')
                if not s_value or s_value == '—': 
                    value = 0.0
                else:
                    value = float(s_value)
            except ValueError:
                return {"value": f"错误: '{original_name}' 的数据 '{value}' 无法转换为数字进行比较。"}
    if pd.isna(value):
        return {"value": f"错误: '{original_name}' 的数据为空 (NaN)。"}
    print(f"--- [实时命中] 成功从 {market_key} 的 '{found_col_name}' 列获取到数据！---")
    return {
        "source": "akshare_spot_api", 
        "identifier": original_name,
        "column": column_label,
        "value": value, 
        "datetime": str(date_time_str)
    }

def get_index_history(
    identifier: str, 
    column_label: str,
    query_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    market: Optional[str] = None,
    original_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取指数的【历史】行情数据。
    """
    def _fail_hist(error_msg: str) -> Dict[str, Any]:
        return {
            "value": f"错误: {error_msg}", 
            "date": datetime.now().strftime('%Y-%m-%d')
        }
    if not query_date:
        return _fail_hist("错误: 历史查询必须提供 `query_date` (单点查询)。")
    if start_date or end_date:
        return _fail_hist("错误: 此函数只支持 `query_date` (单点查询)。请移除 `start_date` 和 `end_date`。")
    entity_info = find_index_code_and_market(identifier=identifier, market_hint=market) 
    if not entity_info:
        return _fail_hist(f"实体链接失败，未能找到与 '{identifier}' 相关的信息。")
    code_for_api, name_for_api, identified_market = entity_info
    market_for_api = identified_market.replace("_index", "")
    effective_query_date_str = query_date
    try:
        for i in range(7): # 7 天回溯
            current_date = datetime.strptime(effective_query_date_str, '%Y-%m-%d') - timedelta(days=i)
            target_date_pd = pd.to_datetime(current_date.strftime('%Y-%m-%d'))
            api_sources = []
            if market_for_api == 'us':
                api_sources = [
                    ("ak.index_us_stock_sina (Sina)", lambda: ak.index_us_stock_sina(symbol=code_for_api))
                ]
            elif market_for_api == 'a':
                api_sources = [
                    ("ak.stock_zh_index_daily_em (Eastmoney)", lambda: ak.stock_zh_index_daily_em(symbol=code_for_api)),
                    ("ak.stock_zh_index_daily (Sina)", lambda: ak.stock_zh_index_daily(symbol=code_for_api)),
                    ("ak.stock_zh_index_daily_tx (Tencent)", lambda: ak.stock_zh_index_daily_tx(symbol=code_for_api))
                ]
            elif market_for_api == 'hk':
                api_sources = [
                    ("ak.stock_hk_index_daily_em (Eastmoney)", lambda: ak.stock_hk_index_daily_em(symbol=code_for_api)),
                    ("ak.stock_hk_index_daily_sina (Sina)", lambda: ak.stock_hk_index_daily_sina(symbol=code_for_api))
                ]
            elif market_for_api == 'global':
                api_sources = [
                    ("ak.index_global_hist_em (Eastmoney)", lambda: ak.index_global_hist_em(symbol=name_for_api)),
                    ("ak.index_global_hist_sina (Sina)", lambda: ak.index_global_hist_sina(symbol=name_for_api))
                ]
            else:
                return _fail_hist(f"未知的市场类型 '{market_for_api}'。")
            all_history_df = None
            source_succeeded = ""
            for source_name, source_func in api_sources:
                try:
                    time.sleep(30)
                    all_history_df = source_func()
                    if all_history_df is not None and not all_history_df.empty:
                        source_succeeded = source_name
                        print(f"--- [历史API] 成功从 {source_name} 获取数据。")
                        break
                except Exception as api_e:
                    print(f"--- [历史API] {source_name} 调用失败: {api_e}")
                    pass
            if all_history_df is not None and not all_history_df.empty:
                all_history_df.rename(columns={
                    'date': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                    **_COLUMN_MAP_AK_TO_STD
                }, inplace=True, errors='ignore')
                if 'date' not in all_history_df.columns: 
                    print(f"--- [历史API] 错误: {source_succeeded} 返回的数据缺少 'date' 列。")
                    continue
                all_history_df['date'] = pd.to_datetime(all_history_df['date'], errors='coerce')
                all_history_df.dropna(subset=['date'], inplace=True)
                hist_df = all_history_df[all_history_df['date'] == target_date_pd]
                if not hist_df.empty:
                    stock_row = hist_df.iloc[0]
                    clean_column_label = column_label.lower().strip() if column_label else 'close'
                    hist_col = clean_column_label
                    if hist_col in stock_row and pd.notna(stock_row[hist_col]):
                            value = stock_row[hist_col]
                            if isinstance(value, str):
                                try:
                                    s_value = value.strip().rstrip('%').replace('+', '').replace(',', '')
                                    value = float(s_value)
                                except (ValueError, TypeError):
                                    return _fail_hist(f"历史数据 '{value}' 无法转换为数字。")
                            actual_date = stock_row.get('date').strftime('%Y-%m-%d')
                            return {
                                "source": "akshare_history_api", 
                                "identifier": name_for_api, 
                                "column": clean_column_label, 
                                "value": float(value),
                                "date": actual_date
                            }
                    else:
                        print(f"--- [历史API] 在 {target_date_pd} 找到了数据，但 '{hist_col}' 列无效或为空。正在回溯...")
                        continue 
        return _fail_hist(f"在日期 '{effective_query_date_str}' 及其前7天内，均未找到 '{identifier}' 的有效历史数据。")
    except Exception as e:
        return _fail_hist(f"在为'{identifier}'获取'{query_date}'的历史数据时发生意外的程序错误: {e}")
    return _fail_hist("未知的函数逻辑错误。")

def _parse_index_price_from_output(output: Any) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(output, dict):
        print(f"  -> [指数解析器] 失败: 底层工具返回的不是字典: {output}")
        return None, None
    price = output.get('value')
    date_str = output.get('datetime', output.get('date'))
    if price is None:
        print(f"  -> [指数解析器] 失败: 未能在 {output} 中找到 'value' 键。")
        return None, date_str
    if date_str is None:
        print(f"  -> [指数解析器] 警告: 未能在 {output} 中找到 'datetime' 或 'date' 键。")
    try:
        if isinstance(price, str):
            print(f"  -> [指数解析器] 失败: 'value' 是一个错误字符串: {price}")
            return None, date_str
        return float(price), date_str
    except (ValueError, TypeError):
        print(f"  -> [指数解析器] 失败: 'value' 不是一个有效的数字: {price}")
        return None, date_str
    
def calculate_index_change_pct(
    identifier: str,
    market: str,
    query_date: str = "latest"
) -> Dict[str, Any]: 
    """
    计算指数在特定日期的涨跌幅。
    """
    def _fail(error_msg: str) -> Dict[str, Any]:
        return {
            "result": { 
                "percentage_change": error_msg
            }
        }
    is_latest_query = (query_date and query_date.lower().strip() in LATEST_KEYWORDS)
    print(f"--- 正在获取 {identifier} (T日) 的收盘价... ---")
    if is_latest_query:
        current_close_output = get_index_realtime(
            identifier=identifier,
            column_label='close',
            market=market
        )
    else:
        current_close_output = get_index_history(
            identifier=identifier,
            column_label='close',
            query_date=query_date,
            market=market
        )
    current_close_price, actual_date_str = _parse_index_price_from_output(current_close_output)
    if current_close_price is None:
        return _fail(f"错误: 无法获取 {identifier} (T日) 的收盘价。底层工具返回: {current_close_output.get('value')}")
    if actual_date_str is None:
        return _fail(f"错误: 无法从 '{current_close_output}' 中解析出实际数据日期(T日)。")
    date_part_only = actual_date_str.split(' ')[0]
    try:
        prev_date_dt = datetime.strptime(date_part_only, '%Y-%m-%d') - timedelta(days=1)
        prev_date_str = prev_date_dt.strftime('%Y-%m-%d')
    except ValueError as e:
        return _fail(f"错误: 无法解析T日日期 '{date_part_only}': {e}")
    print(f"--- 正在查找 {actual_date_str} 前一交易日 (T-1, 即 {prev_date_str}) 的收盘价... ---")
    previous_close_output = get_index_history(
        identifier=identifier,
        column_label='close',
        query_date=prev_date_str, 
        market=market
    )
    previous_close_price, prev_actual_date_str = _parse_index_price_from_output(previous_close_output)
    if previous_close_price is None:
        return _fail(f"错误: 无法获取 {actual_date_str} 之前交易日(T-1)的收盘价。底层工具返回: {previous_close_output.get('value')}")
    if prev_actual_date_str is None:
        prev_actual_date_str = "未知" 
    try:
        if previous_close_price == 0:
            return _fail(f"错误: Calculation Error, T-1日收盘价为0，无法计算涨跌幅。")
        
        price_change_value = current_close_price - previous_close_price
        price_change_pct = (price_change_value / previous_close_price) * 100
        result_json = {
            "result": { 
                "price_change": f"{price_change_value:+.2f}",
                "percentage_change": f"{price_change_pct:+.2f}%" 
            },
            "analysis_type": "index_change_percentage",
            "identifier": identifier,
            "market": market,
            "current_day": {
                "date": actual_date_str,
                "close_price": f"{current_close_price:.2f}"
            },
            "previous_day": {
                "date": prev_actual_date_str,
                "close_price": f"{previous_close_price:.2f}"
            }
        }
        return result_json
    except Exception as e:
        return _fail(f"错误：在计算涨跌幅时发生未知异常: {e}")
    
def _parse_price_from_tool_output(output: str) -> Optional[float]:
    """辅助函数：从 '值 (数据日期: ...)' 格式的字符串中安全地提取价格浮点数。"""
    if not isinstance(output, str):
        return None
    match = re.match(r'^(-?\d+\.?\d*)', output)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            return None
    return None

def _parse_price_and_date_from_output(
    price_output: Dict[str, Any]
) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(price_output, dict):
        print(f"  -> [解析器] 失败: 底层工具返回的不是字典: {price_output}")
        return None, None
    value = None
    raw_value = None 
    try:
        if 'requested_item' in price_output and isinstance(price_output['requested_item'], dict):
            raw_value = price_output['requested_item'].get('value')
        elif 'value' in price_output:
            raw_value = price_output.get('value')
        elif 'result' in price_output:
            raw_value = price_output.get('result')
        else:
            print(f"  -> [解析器] 失败: 找不到 'requested_item', 'value', 或 'result' 键。")
            return None, None
        if raw_value is None:
            print(f"  -> [解析器] 失败: 'value' 键为 None。")
            return None, None
        value = float(raw_value)
    except (ValueError, TypeError) as e:
        print(f"  -> [解析器] 失败: 'value' 无法转换为 float: {e}。原始值: {raw_value}")
        return None, None
    except Exception as e_ex:
        print(f"  -> [解析器] 发生意外错误: {e_ex}")
        return None, None
    date_str = price_output.get('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return value, str(date_str)

def calculate_price_change_pct(
    query_date: str,
    market: str,
    adjust: str = 'qfq',
    name: Optional[str] = None,
    code: Optional[str] = None,
    original_query: Optional[str] = None
) -> Dict[str, Any]: 
    """
    计算单只股票在特定日期的涨跌幅。
    """
    def _fail(error_msg: str) -> Dict[str, Any]:
        return {
            "calculation_result": {
                "percentage_change": error_msg
            }
        }
    if not code and not name:
        return _fail("错误：必须提供股票代码 (code) 或股票名称 (name)。")
    effective_code = code
    if not effective_code and name:
        print(f"--- [代码解析] 缺少代码，正在尝试通过名称 '{name}' 查找代码... ---")
        try:
            resolved_code = get_code_from_name(name=name, market=market) 
            if not resolved_code or isinstance(resolved_code, str) and "--- [LocalSearch] 查找" in resolved_code:
                return _fail(f"错误：get_code_from_name 未能解析 '{name}'。返回: {resolved_code}")
            effective_code = resolved_code 
            print(f"--- [代码解析] 成功找到代码: {effective_code} ---")
        except Exception as e:
            return _fail(f"错误：在为 '{name}' 解析代码时失败: {e}")
    if not effective_code:
        return _fail("错误：最终未能获得一个有效的股票代码用于查询。")
    price_fetcher_map = {'a': get_a_stock_daily_price, 'hk': get_hk_stock_daily_price, 'us': get_us_stock_daily_price}
    if market not in price_fetcher_map:
        return _fail(f"错误：无效的市场类型 '{market}'。支持的市场: 'a', 'hk', 'us'。")
    price_fetcher = price_fetcher_map[market]
    effective_query_date = query_date
    if original_query and any(keyword in original_query.lower() for keyword in LATEST_KEYWORDS):
        today_str = datetime.now().strftime('%Y-%m-%d')
        if query_date != today_str:
            print(f"--- [意图感知] 检测到关键词。将忽略'{query_date}'，强制使用今天'{today_str}'查询。---")
        effective_query_date = today_str
    if query_date.lower().strip() in LATEST_KEYWORDS:
        effective_query_date = 'latest' 
    print(f"--- 正在获取 {effective_query_date} (T日) 的收盘价... ---")
    current_close_output = price_fetcher(
        query_date=effective_query_date, column_label='close', adjust=adjust,
        name=name, code=effective_code, original_query=original_query
    )
    current_close_price, actual_date_str = _parse_price_and_date_from_output(current_close_output)
    if current_close_price is None:
        return _fail(f"错误: 无法获取 {effective_query_date} 的收盘价。底层工具返回: {current_close_output}")
    if actual_date_str is None:
        return _fail(f"错误: 无法从 '{current_close_output}' 中解析出实际数据日期。")
    date_part_only = actual_date_str.split(' ')[0]
    try:
        prev_date_dt = datetime.strptime(date_part_only, '%Y-%m-%d') - timedelta(days=1)
    except ValueError as e:
        return _fail(f"错误: 无法解析T日日期 '{date_part_only}': {e}")
    print(f"--- 正在查找 {actual_date_str} 前一交易日 (T-1) 的收盘价... ---")
    previous_close_output = price_fetcher(
        query_date=prev_date_dt.strftime('%Y-%m-%d'), column_label='close', adjust=adjust,
        name=name, code=effective_code
    )
    previous_close_price, prev_actual_date_str = _parse_price_and_date_from_output(previous_close_output)
    if previous_close_price is None:
        return _fail(f"错误: 无法获取 {actual_date_str} 之前交易日的收盘价。底层工具返回: {previous_close_output}")
    if prev_actual_date_str is None:
        prev_actual_date_str = "未知" 
    try:
        if previous_close_price == 0:
            return _fail(f"错误: Calculation Error, 前一交易日收盘价为0，无法计算涨跌幅。")
        price_change_value = current_close_price - previous_close_price
        price_change_pct = (price_change_value / previous_close_price) * 100
        result_json = {
            "analysis_type": "stock_change_percentage",
            "stock_identifier": name or effective_code,
            "market": market,
            "query_date": query_date,
            "current_day": {
                "date": actual_date_str,
                "close_price": f"{current_close_price:.2f}"
            },
            "previous_day": {
                "date": prev_actual_date_str,
                "close_price": f"{previous_close_price:.2f}"
            },
            "calculation_result": {
                "price_change": f"{price_change_value:+.2f}",
                "percentage_change": f"{price_change_pct:+.2f}%" 
            }
        }
        return result_json
    except Exception as e:
        return _fail(f"错误：在计算涨跌幅时发生未知异常: {e}")
    
_a_history_cache: Dict[tuple, pd.DataFrame] = {}

def calculate_stock_price_range_in_period(
    start_date: str,
    end_date: str,
    adjust: str,
    name: Optional[str] = None,
    code: Optional[str] = None
) -> str:
    """
    计算A股某只股票在【指定时间段内】的最高价与最低价之差。
    """
    if adjust not in ['', 'qfq', 'hfq']:
        return f"错误: 'adjust' 参数 '{adjust}' 无效。有效选项: '', 'qfq', 'hfq'。"
    if not code and not name:
        return "错误：必须提供股票代码 (code) 或股票名称 (name)。"
    try:
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_date_ak = start_date_dt.strftime('%Y%m%d')
        end_date_ak = end_date_dt.strftime('%Y%m%d')
    except ValueError:
        return f"错误: 日期格式无效。请确保 start_date 和 end_date 均为 'YYYY-MM-DD' 格式。"
    symbol = code if code else get_code_from_name(name, market='a')
    if not symbol:
        return f"错误：未能通过名称 '{name}' 或代码 '{code}' 找到对应的A股代码。"
    try:
        print(f"--- 正在为代码'{symbol}'下载从 {start_date} 到 {end_date} 的【A股】历史数据... ---")
        hist_df = ak.stock_zh_a_hist_tx(
            symbol=symbol,
            adjust=adjust,
            start_date=start_date_ak,
            end_date=end_date_ak
        )
        if hist_df is None or hist_df.empty:
            return f"错误: 在指定时间段内未能获取到代码'{symbol}'的任何交易数据。请检查代码或日期范围。"
    except Exception as e:
        return f"下载代码'{symbol}'的历史数据时发生错误: {e}"
    try:
        period_high = hist_df['high'].max()
        period_low = hist_df['low'].min()
        price_difference = period_high - period_low
        result_json = {
            "analysis_type": "stock_price_range",
            "stock_identifier": name or symbol,
            "query_period_start": start_date,
            "query_period_end": end_date,
            "adjust_type": adjust if adjust else "non-adjusted",
            "calculation_result": {
                "period_high": f"{period_high:.2f}",
                "period_low": f"{period_low:.2f}",
                "price_difference": f"{price_difference:.2f}"
            }
        }
        return result_json
    except KeyError as ke:
        return f"错误：返回的数据中缺少必需的列: {ke}。无法进行计算。"
    except Exception as e:
        return f"计算价格差时发生错误: {e}"

def convert_currency_amount(
    base_currency: str,
    target_currency: str,
    amount: float
) -> Union[float, str]:
    try:
        amount_str = str(amount)
        convert_df = ak.currency_convert(
            base=base_currency,
            to=target_currency,
            amount=amount_str,
            api_key=CURRENCY_API_KEY
        )
        if convert_df.empty:
            return "错误: API 未能返回换算结果。"
        convert_df.set_index('item', inplace=True)
        converted_value = float(convert_df.loc['value', 'value'])
        exchange_rate = float(convert_df.loc['rate', 'value'])
        last_updated_ts = int(convert_df.loc['updated', 'value'])
        last_updated_time = datetime.fromtimestamp(last_updated_ts).isoformat()
        result_json = {
            "base_currency": base_currency,
            "target_currency": target_currency,
            "original_amount": amount,
            "converted_amount": f"{converted_value:.4f}",
            "exchange_rate": exchange_rate,
            "rate_last_updated": last_updated_time
        }
        return result_json
    except KeyError:
        return f"错误: API 返回的数据格式不正确，无法找到换算结果 'value'。"
    except Exception as e:
        return f"换算时发生错误: {e}"

_a_dividend_payout_cache: Dict[str, pd.DataFrame] = {}
def get_a_dividend_payout(
    report_date: str,
    name: Optional[str] = None,
    code: Optional[str] = None
) -> Optional[str]:
    """
    查询并获取A股上市公司在特定报告期的【分红方案说明】。

    Args:
        report_date (str): 查询的报告期，格式如 "2023年报", "2024中报"。
        name (Optional[str]): 股票的中文名称。
        code (Optional[str]): 股票的6位数字代码。

    Returns:
        Optional[str]: 查询到的"分红方案说明"文本，如果查询失败则返回 None。
    """
    if not code and not name:
        print("错误：必须提供股票代码 (code) 或股票名称 (name)。")
        return None
    symbol_with_prefix = code if code else get_code_from_name(name)
    if not symbol_with_prefix:
        print(f"错误：未能通过名称 '{name}' 找到对应的股票代码。")
        return None
    symbol = re.sub(r'^[a-zA-Z]+', '', symbol_with_prefix)
    try:
        if symbol not in _a_dividend_payout_cache:
            print(f"缓存未命中，为代码'{symbol}'下载分红派息记录...")
            payout_df = ak.stock_fhps_detail_ths(symbol=symbol)
            if payout_df is None or payout_df.empty:
                print(f"错误：未能获取到代码 '{symbol}' 的分红记录。")
                _a_dividend_payout_cache[symbol] = pd.DataFrame()
                return None
            _a_dividend_payout_cache[symbol] = payout_df
            print("分红记录缓存成功。")
        df = _a_dividend_payout_cache[symbol]
        if df.empty:
            return None
        result_row = df[df['报告期'] == report_date]
        if result_row.empty:
            print(f"查询失败: 未找到报告期为 '{report_date}' 的分红记录。")
            return None
        dividend_description = str(result_row.iloc[0]["分红方案说明"])
        implementation_date = str(result_row.iloc[0].get("实施日期", "N/A")) 
        ex_dividend_date = str(result_row.iloc[0].get("除权除息日", "N/A"))
        result_json = {
            "stock_identifier": name or symbol_with_prefix,
            "report_date": report_date,
            "dividend_plan": {
                "description": dividend_description,
                "implementation_date": implementation_date,
                "ex_dividend_date": ex_dividend_date
            }
        }
        return result_json
    except KeyError:
        print(f"内部错误：预期的列 '分红方案说明' 在数据源中不存在。可用字段: {df.columns.tolist()}")
        return None
    except Exception as e:
        print(f"获取或处理数据时发生严重错误: {e}")
        return None

CACHE_TTL_SECONDS = 60
_spot_market_cache: Dict[str, Optional[pd.DataFrame]] = {}

def get_a_stock_market_cap_top_n(
    market: Literal["all", "sh", "sz", "bj"],
    n: int,
    cap_type: Literal["total", "circulating"],
    include_prices: bool = False  
) -> str:
    """
    查询A股指定市场中，按市值排名前 N 的股票列表。
    可选择性地在结果中包含最新价和开盘价。
    """
    global _spot_market_cache
    if not isinstance(n, int) or n <= 0: return f"错误: 'n' 参数必须是一个正整数。"
    api_map = {"all": ak.stock_zh_a_spot_em, "sh": ak.stock_sh_a_spot_em, "sz": ak.stock_sz_a_spot_em, "bj": ak.stock_bj_a_spot_em}
    cap_column_map = {"total": "总市值", "circulating": "流通市值"}
    sort_column = cap_column_map[cap_type]
    market_name_map = {"all": "沪深京A股", "sh": "沪市", "sz": "深市", "bj": "京市"}
    market_display_name = market_name_map[market]
    df = pd.DataFrame()
    current_time = pd.Timestamp.now().timestamp()
    if market in _spot_market_cache and (current_time - _spot_market_cache[market][1]) < CACHE_TTL_SECONDS:
        df = _spot_market_cache[market][0]
        print(f"--- [函数缓存] 成功从缓存中读取 '{market_display_name}' 实时数据。 ---")
    else:
        print(f"--- [API 调用] 正在通过 akshare 下载 '{market_display_name}' 实时数据... ---")
        try:
            df = api_map[market]()
            if df.empty: return f"错误: 从接口获取 '{market_display_name}' 数据失败，返回为空。"
            _spot_market_cache[market] = (df, current_time)
        except Exception as e: return f"错误: 调用 akshare 接口获取 '{market_display_name}' 数据时失败: {e}"
    try:
        if sort_column not in df.columns: return f"错误: 数据源中缺少用于排序的列 '{sort_column}'。"
        top_n_df = df.sort_values(by=sort_column, ascending=False, na_position='last').head(n)
        if top_n_df.empty: return f"在 '{market_display_name}' 市场中未能找到任何有效的股票数据进行排名。"
    except Exception as e: return f"处理数据排序时发生错误: {e}"
    top_stocks_list = []
    for index, row in top_n_df.iterrows():
        market_value = row[sort_column]
        market_value_in_billion = market_value / 1_0000_0000
        stock_info = {
            "rank": index + 1,
            "code": row.get('代码'),
            "name": row.get('名称'),
            "market_cap": f"{market_value_in_billion:,.2f} 亿元"
        }
        if include_prices:
            stock_info["latest_price"] = row.get('最新价', 'N/A')
            stock_info["open_price"] = row.get('今开', 'N/A')
            
        top_stocks_list.append(stock_info)
    result_json = {
        "analysis_type": "market_cap_ranking",
        "market": market_display_name,
        "ranking_basis": sort_column,
        "top_n": n,
        "ranking_results": top_stocks_list 
    }
    return result_json

_balance_sheet_cache: Dict[str, pd.DataFrame] = {}

def get_balance_sheet(date: str, item_name: str, name: Optional[str] = None, code: Optional[str] = None) -> Optional[Any]:
    """
    查询A股公司资产负债表数据。
    """
    global _balance_sheet_cache
    symbol = code if code else get_code_from_name(name)
    if not symbol: return f"错误: 无法找到代码 for '{name or code}'."
    match = re.search(r'\d{6}', str(symbol))
    if match:
        symbol = match.group(0)
    else:
        symbol = str(symbol)
    if not (isinstance(symbol, str) and symbol.isdigit() and len(symbol) == 6):
        return f"错误: 工具 'get_balance_sheet' 仅适用于中国A股（6位数字代码）。'{symbol}' 似乎不是一个有效的A股代码。"
    formatted_date = date.replace('-', '')
    try:
        if formatted_date not in _balance_sheet_cache:
            print(f"缓存未命中，为报告期'{formatted_date}'下载资产负债表...")
            df = None 
            try:
                if re.match(r'^[48]', symbol):
                    df = ak.stock_zcfz_bj_em(date=formatted_date)
                else:
                    df = ak.stock_zcfz_em(date=formatted_date)
            except Exception as ak_error:
                print(f"警告: 调用akshare接口时直接发生错误: {ak_error}")
            if df is None or df.empty or '股票代码' not in df.columns:
                print(f"警告: akshare接口在查询日期'{formatted_date}'时返回了无效数据(None, empty, or missing key columns)。")
            else:
                _balance_sheet_cache[formatted_date] = df
                print("数据缓存成功。")
        df = _balance_sheet_cache[formatted_date]
        if df.empty:
            return f"错误: 未能获取报告期'{formatted_date}'的资产负债表数据，该日期可能非财报日或无数据。"
        stock_row = df[df['股票代码'] == symbol]
        if stock_row.empty:
            return f"查询失败: 在报告期'{formatted_date}'内未找到股票'{symbol}'的数据。"
        if item_name not in stock_row.columns:
            available_cols_sample = ", ".join(stock_row.columns[:5].tolist())
            return f"错误: 指标 '{item_name}' 不存在于财报中。可用指标示例: {available_cols_sample}..."
        value = stock_row.iloc[0][item_name]
        if pd.isna(value):
            value_display = None
        else:
            value_display = value
        result_json = {
            "stock_code": symbol,
            "stock_name": str(stock_row.iloc[0].get("股票简称", name)), 
            "report_date": date, 
            "financial_statement": "Balance Sheet",
            "item_name": item_name,
            "value": value_display,
            "unit": "元"
        }
        return result_json
    except Exception as e:
        return f"查询资产负债表时发生意外错误: {e}"

_car_market_cache: Dict[tuple, pd.DataFrame] = {}

def get_china_car_sales_cpca(
    market_type: Literal["new_energy_vehicle", "passenger_car", "commercial_vehicle"],
    metric_type: Literal["sales", "production", "wholesale", "export"],
    query_month: str,
    query_year: Optional[int] = None
) -> str:
    """
    从乘联会(CPCA)查询中国汽车市场的月度销量/产量数据。
    可查询新能源市场或乘用车/商用车市场的具体指标。
    """
    global _car_market_cache
    cache_key = (market_type, metric_type)
    ak_params = {}
    api_func = None
    market_display_name = ""
    if market_type == "new_energy_vehicle":
        if metric_type != "sales":
            return f"错误: 当 market_type 为 'new_energy_vehicle' 时，metric_type 只能是 'sales'。"
        api_func = ak.car_market_fuel_cpca
        ak_params['symbol'] = "整体市场"
        market_display_name = "新能源汽车"
    else:
        api_func = ak.car_market_total_cpca
        symbol_map = {"passenger_car": "狭义乘用车", "commercial_vehicle": "广义乘用车"}
        indicator_map = {"sales": "零售", "production": "产量", "wholesale": "批发", "export": "出口"}
        ak_params['symbol'] = symbol_map.get(market_type)
        ak_params['indicator'] = indicator_map.get(metric_type)
        market_display_name = ak_params['symbol']
    try:
        if cache_key in _car_market_cache:
            df = _car_market_cache[cache_key]
            print(f"--- [函数缓存] 成功从缓存中读取数据: {cache_key} ---")
        else:
            print(f"--- [API 调用] 缓存未命中，正在下载数据: {cache_key} ---")
            df = api_func(**ak_params)
            if df.empty:
                return f"错误: 从接口获取 {cache_key} 数据失败，返回为空。"
            _car_market_cache[cache_key] = df
    except Exception as e:
        return f"错误: 调用 akshare 接口或处理数据时失败: {e}"
    try:
        year_columns = [col for col in df.columns if '年' in col]
        if not year_columns:
            return "错误: 未在返回的数据中找到年份列。"
        target_year_col = ""
        if query_year:
            target_year_col = f"{query_year}年"
        else:
            year_columns.sort()
            target_year_col = year_columns[-1]
        if target_year_col not in df.columns:
            return f"错误: 未找到年份 '{query_year}' 的数据。可用年份: {[col.replace('年','') for col in year_columns]}"
        month_match_str = query_month.replace('份', '').replace('月', '') + '月'
        target_row = df[df['月份'] == month_match_str]
        if target_row.empty:
            available_months = df['月份'].unique().tolist()
            return f"错误: 未找到月份 '{query_month}'。可用月份: {available_months}"
        value = target_row.iloc[0][target_year_col]
        if pd.isna(value):
            return f"在 {target_year_col} {month_match_str} 找到了记录，但数值为空（可能尚未公布）。"
        result_json = {
            "source": "乘联会(CPCA)",
            "market_type": market_display_name,
            "metric_type": metric_type,
            "time_period": {
                "year": int(target_year_col.replace('年','')),
                "month": month_match_str
            },
            "value": f"{value:,.2f}",
            "unit": "万辆"
        }
        return result_json
    except Exception as e:
        return f"查询时发生未知错误: {e}"

_api_cache = {}

def get_china_macro_pro(
    indicator_name: Literal[
        "cpi", "fiscal_revenue", "money_supply", 
        "gold_forex_reserves", "central_bank_balance_sheet"
    ],
    target_time: Optional[str] = None,
    target_metric: Optional[str] = None
) -> str:
    """
    获取中国的核心宏观经济指标。
    """
    global _api_cache
    
    try:
        if indicator_name in _api_cache:
            df = _api_cache[indicator_name]
            print(f"--- [函数缓存] 成功从缓存中读取 '{indicator_name}' 数据。 ---")
        else:
            api_map = {
                "cpi": ak.macro_china_cpi,
                "fiscal_revenue": ak.macro_china_czsr,
                "money_supply": ak.macro_china_money_supply,
                "gold_forex_reserves": ak.macro_china_foreign_exchange_gold,
                "central_bank_balance_sheet": ak.macro_china_central_bank_balance,
            }
            if indicator_name not in api_map:
                return f"错误：无效的指标名称 '{indicator_name}'。可用选项为: {list(api_map.keys())}"
            print(f"--- [API 调用] 缓存未命中，正在通过 akshare 下载 '{indicator_name}' 数据... ---")
            df: pd.DataFrame = api_map[indicator_name]()
            if df.empty:
                return f"查询 '{indicator_name}' 成功，但接口未返回任何数据。"
            _api_cache[indicator_name] = df # 存入缓存
        time_col = '月份' if '月份' in df.columns else '统计时间'
        df_indexed = df.set_index(time_col, inplace=False)
        if not target_time and not target_metric:
            latest_time = df_indexed.index[0]
            available_metrics = df_indexed.columns.tolist()
            summary_json = {
                "summary": f"'{indicator_name}' 数据查询成功。",
                "latest_data_time": latest_time,
                "available_metrics": available_metrics,
                "guidance": "请提供 'target_time' 和 'target_metric' 以获取具体数值。"
            }
            return summary_json
        if not target_time or not target_metric:
            return "错误：当查询具体数值时，必须同时提供 'target_time' 和 'target_metric' 两个参数。"
        clean_time_input = target_time.strip().lower()
        query_time_str = None
        if clean_time_input == 'latest':
            query_time_str = df_indexed.index[0]
        else:
            match = re.match(r"(\d{4})[-年]?(\d{1,2})", clean_time_input)
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                index_example = df_indexed.index[0]
                if '月份' in index_example:
                    query_time_str = f"{year}年{month:02d}月份"
                elif '.' in index_example:
                    query_time_str = f"{year}.{month:02d}"
                else: query_time_str = f"{year}-{month:02d}"
        if not query_time_str:
            query_time_str = target_time
        try:
            clean_metric = target_metric.strip()
            
            if query_time_str not in df_indexed.index:
                raise KeyError(f"时间点 '{query_time_str}' 不存在。")
            if clean_metric not in df_indexed.columns:
                raise KeyError(f"指标 '{clean_metric}' 无效。")
            value = df_indexed.loc[query_time_str, clean_metric]
            if pd.isna(value):
                return f"在 '{query_time_str}' 找到了指标 '{clean_metric}'，但其数值为空。"
            final_value = value.item() if hasattr(value, 'item') else value
            result_json = {
                "indicator_name": indicator_name,
                "time_period": query_time_str,
                "metric_name": clean_metric,
                "value": final_value
            }
            return result_json
        except KeyError as e:
            available_times = df_indexed.index[:5].tolist()
            available_metrics = df_indexed.columns.tolist()
            return (
                f"错误: {e}\n"
                f"请检查您的输入。\n"
                f"可用的时间格式示例: {available_times}\n"
                f"可用的指标: {available_metrics}"
            )
    except Exception as e:
        return f"查询 '{indicator_name}' 时发生严重错误: {e}"

_population_cache: Optional[pd.DataFrame] = None

def get_china_population_nbs(
    population_metric: Literal["total", "male", "female", "urban", "rural"],
    query_year: str = "latest"
) -> dict: 
    """
    从国家统计局(NBS)查询中国的年度人口数据 (已修复value为整数类型)。
    """
    global _population_cache
    METRIC_TO_ROW_MAP = {
        "total": "年末总人口(万人)",
        "male": "男性人口(万人)",
        "female": "女性人口(万人)",
        "urban": "城镇人口(万人)",
        "rural": "乡村人口(万人)"
    }
    target_row_name = METRIC_TO_ROW_MAP.get(population_metric)
    if not target_row_name:
        return {"error": f"无效的人口指标 '{population_metric}'。"}
    try:
        if _population_cache is None:
            print(f"--- [API 调用] 缓存未命中，正在通过 akshare 下载中国年度人口数据... ---")
            df = ak.macro_china_nbs_nation(kind="年度数据", path="人口 > 总人口", period="2000-")
            if df.empty:
                return {"error": "从国家统计局接口获取人口数据失败，返回为空。"}
            _population_cache = df
        df = _population_cache
        print(f"--- [函数缓存] 成功从缓存中读取人口数据。 ---")
    except Exception as e:
        return {"error": f"调用 akshare 接口或处理数据时失败: {e}"}
    try:
        target_column_name = ""
        if query_year.lower() == "latest":
            target_column_name = df.columns[0]
        else:
            target_column_name = f"{query_year}年"
        if target_column_name not in df.columns:
            available_years = [col.replace('年', '') for col in df.columns]
            return {"error": f"未能找到年份 '{query_year}' 的数据。可用年份示例: {available_years[:5]}..."}
        value = df.loc[target_row_name, target_column_name]
        if pd.isna(value):
            return {"error": f"在 {target_column_name} 找到了指标 '{target_row_name}'，但其数值为空。"}
        year_str = target_column_name.replace('年', '')
        result_json = {
            "source": "国家统计局(NBS)",
            "region": "中国",
            "year": int(year_str),
            "metric_name": target_row_name.split('(')[0],
            "value": int(value), 
            "unit": "万人"
        }
        return result_json
    except KeyError as e:
        return {"error": f"查询数据时发生键错误，可能是指标 '{target_row_name}' 不存在: {e}"}
    except Exception as e:
        return {"error": f"查询时发生未知错误: {e}"}

def get_currency_history_value(
    base: str, 
    date: str, 
    currency: str, 
    column_label: str
) -> Union[str, float, int, None]:
    """
    获取指定基准货币在特定历史日期的汇率数据。
    """

    try:
        currency_history_df = ak.currency_history(
            base=base, 
            date=date, 
            symbols="", 
            api_key=CURRENCY_API_KEY
        )
        if currency_history_df.empty:
            return f"错误: API 未能返回 {date} 的任何历史数据。"
        currency_history_df.set_index('currency', inplace=True)
        value_series = currency_history_df.get(column_label)
        if value_series is None:
            raise KeyError(f"列 '{column_label}' 不存在。")
        value = value_series.get(currency)
        if value is None:
            raise KeyError(f"货币 '{currency}' 不存在。")
        final_value = value.item() if hasattr(value, 'item') else value
        result_json = {
            "base_currency": base,
            "target_currency": currency,
            "date": date,
            "metric_name": column_label,
            "value": final_value
        }
        return result_json
    except KeyError as e:
        return f"错误: 在 {date} 的数据中找不到货币或列。详细信息: {e}"
    except Exception as e:
        return f"查询时发生错误: {e}"
    
def get_current_latest_value(
    base: str, 
    currency: str, 
    column_label: str, 
    symbol: Optional[str] = None
):
    """
    获取指定基准货币的最新汇率数据。
    """
    try:
        currency_latest_df = ak.currency_latest(
            base=base, 
            symbols=symbol, 
            api_key=CURRENCY_API_KEY
        )
        if currency_latest_df is None or currency_latest_df.empty:
            return f"错误: API 未能返回任何最新的汇率数据。"
        currency_latest_df.set_index('currency', inplace=True)
        if currency not in currency_latest_df.index:
            raise KeyError(f"货币 '{currency}' 不在返回的数据中。")
        if column_label not in currency_latest_df.columns:
            raise KeyError(f"列 '{column_label}' 不在返回的数据中。")
        value = currency_latest_df.loc[currency, column_label]
        final_value = value.item() if hasattr(value, 'item') else value
        last_updated_ts = currency_latest_df.loc[currency].get('timestamp')
        if last_updated_ts is not None:
            last_updated_time = datetime.fromtimestamp(int(last_updated_ts)).isoformat()
        else:
            last_updated_time = "N/A"
        result_json = {
            "base_currency": base,
            "target_currency": currency,
            "data_type": "latest_realtime",
            "metric_name": column_label,
            "value": final_value,
            "rate_last_updated": last_updated_time
        }
        return result_json
    except KeyError as e:
        return f"错误: 查询失败。{e}"
    except Exception as e:
        return f"查询时发生未知错误: {e}"

_dividend_allotment_cache: Dict[str, pd.DataFrame] = {}

def get_dividend_allotment_history(
    report_period: str,
    column_label: str,
    name: Optional[str] = None,
    code: Optional[str] = None
) -> Optional[Any]:
    """
    查询A股公司历史分红配送数据。
    """
    global _dividend_allotment_cache
    symbol_from_input = code if code else get_code_from_name(name)
    if not symbol_from_input: 
        return f"错误: 无法找到代码 for '{name or code}'."
    symbol = re.sub(r'\D', '', str(symbol_from_input))
    try:
        if symbol not in _dividend_allotment_cache:
            print(f"缓存未命中，为代码'{symbol}'下载历史分红配送数据...")
            df = ak.stock_fhps_detail_em(symbol=symbol)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df['报告期'] = pd.to_datetime(df['报告期']).dt.strftime('%Y-%m-%d')
                _dividend_allotment_cache[symbol] = df
            else:
                _dividend_allotment_cache[symbol] = None
            print("数据缓存成功。")
        df = _dividend_allotment_cache[symbol]
        if df is None:
            return f"错误: 未能获取代码'{symbol}'的分红配送数据。"
        df_indexed = df.set_index('报告期')
        if column_label not in df_indexed.columns:
            raise KeyError(f"指标 '{column_label}' 无效。")
        if report_period not in df_indexed.index:
            raise KeyError(f"报告期 '{report_period}' 不存在。")
        value = df_indexed.loc[report_period, column_label]
        plan_description = df_indexed.loc[report_period, '分红方案说明'] 
        result_json = {
            "stock_identifier": name or symbol_from_input,
            "report_period": report_period,
            "metric_name": column_label,
            "value": value,
            "dividend_plan_description": plan_description
        }
        return result_json
    except KeyError as e:
        if 'df' in locals() and df is not None:
            available_periods = df['报告期'].unique().tolist()
            available_columns = df.columns.tolist()
            return (f"查询失败: {e}\n"
                    f"请检查您的输入。\n"
                    f"可用报告期示例: {available_periods[:5]}...\n"
                    f"可用指标: {available_columns}")
        return f"查询失败: {e}"
    except Exception as e:
        return f"查询分红配送时出错: {e}"

_earnings_announcement_cache: Dict[str, pd.DataFrame] = {}

def get_earnings_announcement(
    report_date: str,
    item_name: str,
    name: Optional[str] = None,
    code: Optional[str] = None,
) -> Any:
    """
    查询A股公司在特定报告期发布的【业绩预告】。
    """
    global _earnings_announcement_cache
    if not code and not name: return "错误: 必须提供股票代码或名称。"
    symbol_from_input = code if code else get_code_from_name(name)
    if not symbol_from_input: return f"错误: 无法找到代码 for '{name or code}'."
    symbol_code = re.sub(r'\D', '', str(symbol_from_input))
    try:
        formatted_date = pd.to_datetime(report_date).strftime('%Y%m%d')
        if formatted_date not in _earnings_announcement_cache:
            print(f"缓存未命中，下载'{formatted_date}'的所有业绩预告...")
            df = ak.stock_yjyg_em(date=formatted_date)
            _earnings_announcement_cache[formatted_date] = df if isinstance(df, pd.DataFrame) else None
            print("数据缓存成功。")
        df = _earnings_announcement_cache[formatted_date]
        if df is None: return f"错误: 未能获取'{formatted_date}'的业绩预告数据。"
        stock_df = df[df['股票代码'] == symbol_code].copy()
        if stock_df.empty: return f"查询失败: 未找到代码'{symbol_code}'在'{report_date}'的业绩预告。"
        stock_df['公告日期'] = pd.to_datetime(stock_df['公告日期'])
        latest_announcement = stock_df.sort_values(by='公告日期', ascending=False).iloc[0]
        if item_name not in latest_announcement.index:
            raise KeyError
        value = latest_announcement[item_name]
        stock_name_from_data = latest_announcement.get('股票简称', name)
        announcement_date = latest_announcement['公告日期'].strftime('%Y-%m-%d')
        forecast_type = latest_announcement.get('预告类型', 'N/A')
        result_json = {
            "stock_code": symbol_code,
            "stock_name": stock_name_from_data,
            "report_date": report_date,
            "announcement_date": announcement_date, 
            "forecast_type": forecast_type,
            "requested_item": item_name,
            "value": value
        }
        return result_json 
    except KeyError:
        if 'latest_announcement' in locals():
            available_items = latest_announcement.index.tolist()
            return f"查询失败: 预告中不存在'{item_name}'字段。可用字段: {available_items}"
        return f"查询失败: 预告中不存在'{item_name}'字段。"
    except Exception as e:
        return f"查询业绩预告时出错: {e}"

_earnings_report_cache: Dict[str, pd.DataFrame] = {}

def get_earnings_report_summary(date: str, column_label: str, name: Optional[str] = None, code: Optional[str] = None) -> Optional[Any]:
    """
    查询A股公司在特定报告期的业绩报表摘要。
    
    :param date: 要查询的财报报告期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'。
    :param column_label: 要查询的指标名称，例如 "营业总收入-同比增长"。
    :param name: [可选] 股票的中文名称。
    :param code: [可选] 股票的6位数字代码。
    :return: 返回查询到的具体数值或错误信息。
    """
    global _earnings_report_cache
    symbol = code if code else get_code_from_name(name, market='a')
    if not symbol: return f"错误: 无法找到代码 for '{name or code}'."
    symbol = re.sub(r'\D', '', str(symbol))
    if not (isinstance(symbol, str) and len(symbol) == 6):
        return f"错误: 股票标识 '{symbol}' 非标准A股代码格式。"
    formatted_date = date.replace('-', '')
    try:
        if formatted_date not in _earnings_report_cache:
            print(f"缓存未命中，为报告期'{formatted_date}'下载业绩报表...")
            df = ak.stock_yjbb_em(date=formatted_date)
            if df is None or df.empty:
                return f"错误: 未能获取报告期'{formatted_date}'的业绩报表数据。该日期可能非有效财报日。"
            _earnings_report_cache[formatted_date] = df 
            print("数据缓存成功。")
        df = _earnings_report_cache[formatted_date]
        stock_row = df[df['股票代码'] == symbol]
        if stock_row.empty: 
            return f"查询失败: 在报告期'{formatted_date}'的业绩报表中未找到股票'{symbol}'的数据。"
        if column_label not in stock_row.columns:
            available_cols_sample = ", ".join(stock_row.columns.tolist())
            return f"错误: 指标 '{column_label}' 不存在。可用指标包括: {available_cols_sample}。"
        value = float(stock_row.iloc[0][column_label])
        stock_name_from_data = stock_row.iloc[0].get("股票简称", name)
        if pd.isna(value):
            value_display = None
        else:
            if '同比增长' in column_label or '增长率' in column_label:
                value_display = f"{value:.2f}%"
            else:
                value_display = value
        result_json = {
            "stock_code": symbol,
            "stock_name": stock_name_from_data,
            "report_date": date,
            "financial_statement": "Earnings Report Summary",
            "item_name": column_label,
            "value": value_display
        }
        return result_json
    except Exception as e:
        return f"查询业绩报表时出错: {e}"

_financial_indicators_cache: Dict[str, pd.DataFrame] = {}
def get_financial_indicators(
    report_date: str,
    item_name: str,
    name: Optional[str] = None,
    code: Optional[str] = None
) -> Optional[Any]:
    global _financial_indicators_cache
    symbol_from_input = code if code else get_code_from_name(name)
    if not symbol_from_input: return f"错误: 无法找到代码 for '{name or code}'."
    
    symbol = re.sub(r'\D', '', str(symbol_from_input))
    if re.match(r'^6', symbol): prefixed_symbol = f"{symbol}.SH"
    elif re.match(r'^[03]', symbol): prefixed_symbol = f"{symbol}.SZ"
    elif re.match(r'^[48]', symbol): prefixed_symbol = f"{symbol}.BJ"
    else: prefixed_symbol = symbol 
    try:
        if prefixed_symbol not in _financial_indicators_cache:
            print(f"缓存未命中，为代码'{prefixed_symbol}'下载所有历史财务指标...")
            df = ak.stock_financial_analysis_indicator_em(symbol=prefixed_symbol, indicator="按报告期")
            _financial_indicators_cache[prefixed_symbol] = df if isinstance(df, pd.DataFrame) else None
            print("数据缓存成功。")
        df = _financial_indicators_cache[prefixed_symbol]
        if df is None: return f"错误: 未能获取代码'{prefixed_symbol}'的财务指标数据。"
        df_copy = df.copy()
        df_copy['REPORT_DATE'] = pd.to_datetime(df_copy['REPORT_DATE']).dt.strftime('%Y-%-m-%d')
        result_row = df_copy[df_copy['REPORT_DATE'] == report_date]
        if result_row.empty:
            available_dates = df_copy['REPORT_DATE'].unique().tolist()
            return f"查询失败: 未找到报告期为 '{report_date}' 的财务指标数据。可用报告期示例: {available_dates[:5]}..."
        if item_name not in result_row.columns:
            available_items = result_row.columns.tolist()
            return f"查询失败: 指标 '{item_name}' 不存在。可用指标示例: {available_items[:10]}..."
        value = result_row.iloc[0][item_name]
        if pd.isna(value):
            value_display = None
        else:
            value_display = value
        result_json = {
            "stock_identifier": name or symbol_from_input,
            "report_date": report_date,
            "data_source": "Financial Indicators",
            "item_name": item_name,
            "value": value_display
        }
        return result_json
    except Exception as e:
        return f"查询财务指标时出错: {e}"

FOREIGN_FUTURES_ALIAS_MAP: Dict[str, str] = {
    'wti': 'NYMEX原油', 'wticrudeoil': 'NYMEX原油', 'nymexcrudeoil': 'NYMEX原油', 'cl': 'NYMEX原油',
    'brent': '布伦特原油', 'brentcrudeoil': '布伦特原油', 'co': '布伦特原油',
    'gold': 'COMEX黄金', 'comexgold': 'COMEX黄金', 'gc': 'COMEX黄金',
    'silver': 'COMEX白银', 'comexsilver': 'COMEX白银', 'si': 'COMEX白银',
    'COMEX copper': 'COMEX铜', 'naturalgas': 'NYMEX天然气', 'ng': 'NYMEX天然气',
    'lme尼克尔': 'LME镍3个月',
}

def get_foreign_futures_realtime(
    commodity_name: str,
    column_label: str
) -> str:
    """
    【国际期货/实时专用】获取指定国际大宗商品的【最新实时行情数据】。
    """
    try:
        normalized_input = normalize_name(commodity_name)
        target_name = FOREIGN_FUTURES_ALIAS_MAP.get(normalized_input)

        if not target_name:
            return (f"错误：未能识别的国际期货品种 '{commodity_name}'。"
                    f"请尝试使用标准名称，如 'WTI Crude Oil', 'COMEX Gold', 'Brent' 等。")
        print(f"--- [国际期货] 正在获取所有国际期货的实时快照... ---")
        try:
            all_symbols_list = ak.futures_foreign_commodity_subscribe_exchange_symbol()
            if not all_symbols_list:
                return "错误：无法获取国际期货的商品代码列表。"
            realtime_df = ak.futures_foreign_commodity_realtime(symbol=all_symbols_list)
        except Exception as e:
            return f"错误：调用 akshare 底层接口 'futures_foreign_commodity_realtime' 失败: {e}"
        if realtime_df.empty:
            return "错误：调用 futures_foreign_commodity_realtime 接口未能返回任何数据。"
        match_df = realtime_df[realtime_df['名称'] == target_name]
        if match_df.empty:
            available_commodities = realtime_df['名称'].unique().tolist()
            return (f"错误：在实时数据中未找到 '{target_name}' (您查询的是 '{commodity_name}')。\n"
                    f"当前可用的品种列表为: {available_commodities}")
        column_map = {
            'name': '名称', 'close': '最新价', 'price_cny': '人民币报价',
            'change': '涨跌额', 'change_percent': '涨跌幅', 'open': '开盘价',
            'high': '最高价', 'low': '最低价', 'previous_settle': '昨日结算价',
            'open_interest': '持仓量', 'bid': '买价', 'ask': '卖价',
            'time': '行情时间', 'date': '日期',
        }
        if column_label not in column_map:
            return f"错误：列 '{column_label}' 无效。有效列为: {list(column_map.keys())}"
        actual_column = column_map[column_label]
        if actual_column not in match_df.columns:
            return f"错误：数据源中不存在名为 '{actual_column}' 的列。"
        row_data = match_df.iloc[0]
        value = row_data[actual_column]
        result_json = {
            "commodity_name": row_data.get('名称'),
            "data_type": "latest_realtime",
            "requested_item": {
                "label": column_label,
                "value": value
            },
            "realtime_quote": {
                "latest_price": row_data.get('最新价'),
                "price_cny": row_data.get('人民币报价'),
                "change_value": row_data.get('涨跌额'),
                "change_percent": row_data.get('涨跌幅'),
                "open_price": row_data.get('开盘价'),
                "high_price": row_data.get('最高价'),
                "low_price": row_data.get('最低价'),
                "previous_settle": row_data.get('昨日结算价'),
                "quote_time": f"{row_data.get('日期')} {row_data.get('行情时间')}"
            }
        }
        return result_json
    except Exception as e:
        return f"获取国际期货实时数据时发生未知错误: {e}"
    
def get_futures_daily_price(
    symbol: str, 
    start_date: str, 
    end_date: Optional[str] = None
) -> str:
    """
    获取指定期货合约在特定日期范围内的日线行情数据。
    在查询前会严格校验日期是否为有效交易日。
    """
    try:
        if not end_date:
            end_date = start_date
        try:
            recent_trades_df = ak.futures_inventory_em(symbol="螺纹钢")
            if recent_trades_df.empty:
                raise Exception("无法通过东方财富库存接口获取最新的交易日信息。")
            most_recent_trade_date_str = recent_trades_df['日期'].iloc[-1]
            most_recent_trade_date_obj = pd.to_datetime(most_recent_trade_date_str).date()
            request_date_obj = datetime.strptime(start_date, "%Y%m%d").date()
        except ValueError:
            return f"错误：日期格式不正确 '{start_date}'。请使用 'YYYYMMDD' 格式。"
        except Exception as e:
            return f"获取或解析最新交易日时发生错误: {e}"
        if request_date_obj > most_recent_trade_date_obj:
            return (
                f"错误：您请求的日期 '{start_date}' 是一个未来的日期。\n"
                f"指令：请立即放弃当前尝试，并使用已知的最近一个有效交易日 '{most_recent_trade_date_obj.strftime('%Y%m%d')}' 重新发起一次新的查询。"
            )
        markets = ["CFFEX", "SHFE", "DCE", "CZCE", "GFEX", "INE"]
        all_dfs = []
        for market in markets:
            try:
                df_market = ak.get_futures_daily(start_date=start_date, end_date=end_date, market=market)
                all_dfs.append(df_market)
            except Exception:
                continue
        if not all_dfs or pd.concat(all_dfs).empty:
            return f"错误：在有效的日期 '{start_date}' 无法获取任何期货数据。这很可能是一个节假日。\n指令：请更换一个有效的交易日重试。"
        df_all_markets = pd.concat(all_dfs)
        df = df_all_markets[df_all_markets['symbol'] == symbol]
        if not df.empty:
            df_copy = df.copy()
            for col in ['open', 'high', 'low', 'close', 'volume', 'open_interest', 'turnover', 'settle', 'pre_settle']:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            df_copy['date'] = pd.to_datetime(df_copy['date']).dt.strftime('%Y-%m-%d')
            json_result = df_copy.to_json(orient="records", force_ascii=False, indent=2)
            return json_result
        else:
            match = re.match(r'([a-zA-Z]+)', symbol.lower())
            if not match: 
                return f"查询成功，但在有效交易日 '{start_date}' 未找到合约 '{symbol}' 的数据。"
            base_symbol = match.group(1)
            master_list_df = ak.futures_comm_info(symbol="所有")
            commodity_df = master_list_df[master_list_df['合约代码'].str.lower().str.startswith(base_symbol)]
            if commodity_df.empty:
                return f"查询成功，但未找到合约 '{symbol}' 的数据，也未在当前市场中找到任何与品种 '{base_symbol}' 相关的合约。"
            else:
                available_codes = commodity_df['合约代码'].unique().tolist()
                if available_codes:
                    probe_symbol = available_codes[0]
                    probe_df = df_all_markets[df_all_markets['symbol'] == probe_symbol]
                    if probe_df.empty:
                        return (
                            f"错误：在日期 '{start_date}'，您查询的品种 '{base_symbol}' (包括合约 {symbol}, {probe_symbol} 等) 似乎整体休市或所有合约均未开始交易。\n"
                            f"指令：请更换一个有效的交易日重试，或者更换一个商品品种进行查询。"
                        )
                return (
                    f"错误：合约代码 '{symbol}' 在有效的交易日 '{start_date}' 内没有数据。\n"
                    f"原因可能是该合约已退市或尚未上市。指令：请从以下 '{base_symbol}' 品种的可用合约列表中选择一个，并使用相同的有效日期 '{start_date}' 重试本函数。\n"
                    f"可用合约列表: {available_codes}"
                )
    except Exception as e:
        import traceback
        return f"获取期货日线数据时发生未知错误: {e}\n{traceback.format_exc()}"

def get_futures_price_on_date(
    symbol: str, 
    query_date: str
) -> Union[Dict[str, Any], str]:
    """
    获取指定期货合约在【某一个指定日期】的日线行情数据。
    
    :param symbol: 要查询的期货合约代码, 例如 "rb2401"。
    :param query_date: 要查询的日期, 格式 'YYYYMMDD' (注意: 与其他工具不同, 此处遵循akshare API)。
    :return: 一个包含该日所有行情数据的字典, 或一个错误信息字符串。
    """
    print(f"--- [期货单日查询] 正在获取 '{symbol}' 在 {query_date} 的数据... ---")
    try:
        try:
            recent_trades_df = ak.futures_inventory_em(symbol="螺纹钢") 
            if recent_trades_df.empty:
                raise Exception("无法通过东方财富库存接口获取最新的交易日信息。")
            most_recent_trade_date_str = recent_trades_df['日期'].iloc[-1]
            most_recent_trade_date_obj = pd.to_datetime(most_recent_trade_date_str).date()
            request_date_obj = datetime.strptime(query_date, "%Y%m%d").date()
        except ValueError:
            return f"错误：日期格式不正确 '{query_date}'。请使用 'YYYYMMDD' 格式。"
        except Exception as e:
            return f"获取或解析最新交易日时发生错误: {e}"
        if request_date_obj > most_recent_trade_date_obj:
            return (
                f"错误：您请求的日期 '{query_date}' 是一个未来的日期。\n"
                f"指令：请立即放弃当前尝试，并使用已知的最近一个有效交易日 '{most_recent_trade_date_obj.strftime('%Y%m%d')}' 重新发起一次新的查询。"
            )
        markets = ["CFFEX", "SHFE", "DCE", "CZCE", "GFEX", "INE"]
        all_dfs = []
        for market in markets:
            try:
                df_market = ak.get_futures_daily(start_date=query_date, end_date=query_date, market=market)
                all_dfs.append(df_market)
            except Exception:
                continue
        if not all_dfs or pd.concat(all_dfs).empty:
            return f"错误：在有效的日期 '{query_date}' 无法获取任何期货数据。这很可能是一个节假日。\n指令：请更换一个有效的交易日重试。"
        df_all_markets = pd.concat(all_dfs)
        df = df_all_markets[df_all_markets['symbol'] == symbol]
        if not df.empty:
            data_series = df.iloc[0]
            result_dict = data_series.to_dict()
            for col in ['open', 'high', 'low', 'close', 'volume', 'open_interest', 'turnover', 'settle', 'pre_settle']:
                if col in result_dict:
                    result_dict[col] = pd.to_numeric(result_dict[col], errors='coerce')
            result_dict['date'] = pd.to_datetime(result_dict['date']).dt.strftime('%Y-%m-%d')
            return result_dict
        else:
            match = re.match(r'([a-zA-Z]+)', symbol.lower())
            if not match: 
                return f"查询成功，但在有效交易日 '{query_date}' 未找到合约 '{symbol}' 的数据。"
            base_symbol = match.group(1)
            master_list_df = ak.futures_comm_info(symbol="所有")
            commodity_df = master_list_df[master_list_df['合约代码'].str.lower().str.startswith(base_symbol)]
            if commodity_df.empty:
                return f"查询成功，但未找到合约 '{symbol}' 的数据，也未在当前市场中找到任何与品种 '{base_symbol}' 相关的合约。"
            else:
                available_codes = commodity_df['合约代码'].unique().tolist()
                return (
                    f"错误：合约代码 '{symbol}' 在有效的交易日 '{query_date}' 内没有数据。\n"
                    f"原因可能是该合约已退市或尚未上市。指令：请从以下 '{base_symbol}' 品种的可用合约列表中选择一个，并使用相同的有效日期 '{query_date}' 重试本函数。\n"
                    f"可用合约列表: {available_codes}"
                )
    except Exception as e:
        return f"获取期货日线数据时发生未知错误: {e}\n{traceback.format_exc()}"

_fx_spot_quote_cache: Dict[str, Any] = {"data": None, "timestamp": 0}

def get_fx_spot_quote_value(
    currency_pair: str,
    metric_name: str
) -> Optional[float]:
    """
    获取外汇即期报价，支持交叉汇率计算。
    """
    global _fx_spot_quote_cache
    current_time = time.time()
    if (current_time - _fx_spot_quote_cache["timestamp"]) > CACHE_TTL_SECONDS:
        print("缓存已过期或首次查询，正在下载最新的外汇即期报价...")
        try:
            df = ak.fx_spot_quote()
            if df is not None and not df.empty:
                df.set_index('货币对', inplace=True)
                _fx_spot_quote_cache["data"] = df
                _fx_spot_quote_cache["timestamp"] = current_time
        except Exception as e:
            print(f"下载外汇即期报价时出错: {e}")
            _fx_spot_quote_cache["timestamp"] = current_time
            return None
    df = _fx_spot_quote_cache["data"]
    if df is None:
        return None
    try:
        row_data = df.loc[currency_pair]
        value = float(row_data[metric_name])
        result_json = {
            "query_type": "direct_quote",
            "currency_pair": currency_pair,
            "requested_metric": {
                "name": metric_name,
                "value": value
            },
            "full_quote": {
                "bid_price": float(row_data.get('买报价')),
                "ask_price": float(row_data.get('卖报价')),
                "high_price": float(row_data.get('最高')),
                "low_price": float(row_data.get('最低')),
                "open_price": float(row_data.get('今开')),
                "previous_close": float(row_data.get('昨收'))
            }
        }
        return result_json
    except KeyError:
        print(f"直接查询 '{currency_pair}' 失败，尝试计算交叉汇率...")
    try:
        curr1, curr2 = currency_pair.upper().split('/')
        pair1_buy = df.loc[f'{curr1}/CNY', '买报价']
        pair1_sell = df.loc[f'{curr1}/CNY', '卖报价']
        try:
            pair2_buy = df.loc[f'CNY/{curr2}', '买报价']
            pair2_sell = df.loc[f'CNY/{curr2}', '卖报价']
        except KeyError:
            pair2_inverted_buy = df.loc[f'{curr2}/CNY', '买报价']
            pair2_inverted_sell = df.loc[f'{curr2}/CNY', '卖报价']
            pair2_buy = 1 / pair2_inverted_sell
            pair2_sell = 1 / pair2_inverted_buy
        cross_rate_buy = pair1_buy * pair2_buy
        cross_rate_sell = pair1_sell * pair2_sell
        print(f"成功计算交叉汇率 {currency_pair}: 买价={cross_rate_buy}, 卖价={cross_rate_sell}")
        requested_value = None
        if metric_name == '买报价':
            requested_value = float(cross_rate_buy)
        elif metric_name == '卖报价':
            requested_value = float(cross_rate_sell)
        else:
            return f"注意：交叉汇率计算仅支持 '买报价' 和 '卖报价'，不支持 '{metric_name}'。"
        result_json = {
            "query_type": "cross_rate_calculation",
            "currency_pair": currency_pair,
            "requested_metric": {
                "name": metric_name,
                "value": requested_value
            },
            "full_quote": {
                "bid_price": float(cross_rate_buy),
                "ask_price": float(cross_rate_sell)
            }
        }
        return result_json
    except Exception as e:
        return f"计算交叉汇率 '{currency_pair}' 失败: {e},可用货币对包括: {df.index.tolist()}."
    
_global_spot_cache: Optional[pd.DataFrame] = None
_cache_timestamp: float = 0
CACHE_TTL_SECONDS: int = 300  

def normalize_name(name: str) -> str:
    """通用名称标准化函数。"""
    if not isinstance(name, str): return ""
    name = name.lower().strip()
    return re.sub(r'[^\w\u4e00-\u9fa5]', '', name)

INDEX_ALIAS_MAP = {
    # 美国指数
    'sp500': 'SPX', 's&p500': 'SPX', '标普500': 'SPX', 'S&P 500': 'SPX',
    'dowjones': 'DJIA', 'dow': 'DJIA', '道琼斯': 'DJIA',
    'nasdaq': 'NDX', '纳斯达克': 'NDX',
    
    # 亚洲指数
    'hangseng': 'HSI', 'hsi': 'HSI', '恒生指数': 'HSI',
    'nikkei225': 'N225', 'nikkei': 'N225', '日经225': 'N225',
    'kospi': 'KS11', '韩国kospi': 'KS11',
    'jakartacomposite': 'JKSE', 'idxcomposite': 'JKSE', '印尼雅加达综合': 'JKSE',
    
    # 欧洲指数
    'ftse100': 'FTSE', 'uk100': 'FTSE', '富时100': 'FTSE',
    'dax': 'GDAXI', 'dax30': 'GDAXI', '德国dax30': 'GDAXI',
    'cac40': 'FCHI', '法国cac40': 'FCHI', '德国DAX30': 'GDAXI',

    # 其他
    '上证指数': '000001', '沪深300': '000300', '创业板指': '399006',
    'allordinaries': 'AORD'
}

def get_global_index_spot_value(
    name_or_code: str,
    column_label: str,
    force_refresh: bool = False,
    **kwargs 
) -> str:
    """
    通过名称、代码或别名，获取全球主要指数的【最新实时行情数据】。
    """
    if 'date' in kwargs or 'query_date' in kwargs:
        return (
            "错误：参数使用错误。此工具是【实时数据专用工具】，"
            "不接受 'date' 或 'query_date' 等任何日期参数。"
            "如需查询历史数据，请使用其他工具。"
        )
    global _global_spot_cache, _cache_timestamp
    is_cache_stale = (time.time() - _cache_timestamp) > CACHE_TTL_SECONDS
    if force_refresh or _global_spot_cache is None or is_cache_stale:
        if force_refresh: print("--- [实时指数] 已触发缓存强制刷新 (latest/最新)。---")
        if is_cache_stale: print(f"--- [实时指数] 缓存已超过{CACHE_TTL_SECONDS}秒，自动刷新。---")
        print("--- [实时指数] 正在从 akshare 下载全球指数实时快照...")
        try:
            spot_df = ak.index_global_spot_em()
            if spot_df.empty:
                _global_spot_cache, _cache_timestamp = pd.DataFrame(), 0
            else:
                _global_spot_cache, _cache_timestamp = spot_df, time.time()
                print(f"--- [实时指数] 成功缓存 {len(_global_spot_cache)} 条全球指数快照。---")
        except Exception as e:
            _global_spot_cache, _cache_timestamp = pd.DataFrame(), 0
            return f"错误: 调用 ak.index_global_spot_em 接口失败: {e}"
    if _global_spot_cache.empty:
        return "错误: 全球指数实时快照数据当前不可用。"
    df = _global_spot_cache
    normalized_input = normalize_name(name_or_code)
    target_identifier = INDEX_ALIAS_MAP.get(normalized_input, name_or_code)
    df['代码'] = df['代码'].astype(str)
    match = df[df['代码'].str.lower() == target_identifier.lower()]
    if match.empty:
        normalized_df_names = df['名称'].astype(str).apply(normalize_name)
        match = df[normalized_df_names == normalize_name(target_identifier)]
    if match.empty:
        return f"错误: 未能找到名为 '{name_or_code}' 的指数。"
    column_map = {
        'code': '代码', 'name': '名称', 'close': '最新价', 'change': '涨跌额',
        'change_percent': '涨跌幅', 'previous_close': '昨收价', 'amplitude': '振幅',
        'last_update_time': '最新行情时间', 'high': '最高价', 'low': '最低价'
    }
    if column_label not in column_map:
        return f"错误: 列 '{column_label}' 无效。有效列为: {list(column_map.keys())}"
    actual_column = column_map[column_label]
    if actual_column not in df.columns:
        return f"错误: 数据源中不存在名为 '{actual_column}' 的列。"
    row_data = match.iloc[0]
    requested_value = row_data[actual_column]
    result_json = {
        "index_name": row_data.get('名称'),
        "index_code": row_data.get('代码'),
        "data_type": "latest_realtime_spot",
        "requested_item": {
            "label": column_label,
            "value": requested_value
        },
        "full_quote": {
            "latest_price": row_data.get('最新价'),
            "change_value": row_data.get('涨跌额'),
            "change_percent": row_data.get('涨跌幅'),
            "open_price": row_data.get('今开'),
            "high_price": row_data.get('最高'),
            "low_price": row_data.get('最低'),
            "previous_close": row_data.get('昨收'),
            "amplitude": row_data.get('振幅'),
            "quote_time": row_data.get('数据时间')
        }
    }
    return result_json

def get_hk_stock_basic_info(
    item_name: str, 
    name: Optional[str] = None, 
    code: Optional[str] = None
) -> Optional[Any]:
    if not code and not name:
        print("错误：必须提供股票代码 (code) 或股票名称 (name)。")
        return None
    symbol = code if code else get_code_from_name(name)
    if not symbol:
        print(f"错误：未能通过名称 '{name}' 找到对应的股票代码。")
        return None
    try:
        print(f"  -> 正在为代码 '{symbol}' 从雪球查询公司概况...")
        info_df = ak.stock_individual_basic_info_hk_xq(symbol=symbol)
        if info_df is None or info_df.empty:
            print(f"错误：未能获取到代码 '{symbol}' 的基础信息。请检查代码是否有效。")
            return None
        info_df.set_index("item", inplace=True)
        try:
            value = info_df.loc[item_name, 'value']
            stock_code_from_data = info_df.loc['代码', 'value']
            stock_name_from_data = info_df.loc['名称', 'value']
            result_json = {
                "stock_code": stock_code_from_data,
                "stock_name": stock_name_from_data,
                "data_source": "Hong Kong Stock Basic Info",
                "requested_item": {
                    "name": item_name,
                    "value": value
                }
            }
            return result_json
        except KeyError:
            print(f"错误：在 '{symbol}' 的信息中未找到名为 '{item_name}' 的字段。")
            print(f"可用字段包括: {info_df.index.tolist()}") # 打印所有可用的字段名
            return None
    except Exception as e:
        print(f"获取或处理数据时发生严重错误: {e}")
        return None

_income_statement_cache: Dict[str, pd.DataFrame] = {}

def get_income_statement(date: str, item_name: str, name: Optional[str] = None, code: Optional[str] = None) -> Optional[Any]:
    """查询A股公司在特定报告期的【利润表】中的单个科目金额。"""
    global _income_statement_cache
    symbol = code if code else get_code_from_name(name)
    if not symbol: return f"错误: 无法找到代码 for '{name or code}'."
    symbol = re.sub(r'\D', '', str(symbol))
    formatted_date = date.replace('-', '')
    try:
        if formatted_date not in _income_statement_cache:
            print(f"缓存未命中，为报告期'{formatted_date}'下载利润表...")
            df = ak.stock_lrb_em(date=formatted_date)
            _income_statement_cache[formatted_date] = df if not df.empty else pd.DataFrame()
            print("数据缓存成功。")
        df = _income_statement_cache[formatted_date]
        if df.empty: return f"错误: 未能获取报告期'{formatted_date}'的利润表数据。"
        stock_row = df[df['股票代码'] == symbol]
        if stock_row.empty: return f"查询失败: 在该报告期未找到股票'{symbol}'的数据。"
        if item_name not in stock_row.columns:
            available_items = stock_row.columns.tolist()
            return f"查询失败: 指标 '{item_name}' 不存在。可用指标示例: {available_items[:10]}..."
        value = stock_row.iloc[0][item_name]
        stock_name_from_data = stock_row.iloc[0].get("股票简称", name)
        if pd.isna(value):
            value_display = None
        else:
            value_display = value
        result_json = {
            "stock_code": symbol,
            "stock_name": stock_name_from_data,
            "report_date": date,
            "financial_statement": "Income Statement",
            "item_name": item_name,
            "value": value_display,
            "unit": "元" 
        }
        return result_json
    except Exception as e:
        return f"查询利润表时出错: {e}"

def get_nbs_region_data(
    kind: str,
    path: str,
    period: str,
    region: Optional[str] = None,
    indicator: Optional[str] = None,
    target_label: Optional[str] = None,
    target_time: Optional[str] = None
) -> str:
    if not indicator and not region:
        return "错误：参数 'indicator' (指标) 和 'region' (地区) 不能同时为空。"
    try:
        df: pd.DataFrame = ak.macro_china_nbs_region(
            kind=kind, path=path, period=period, indicator=indicator, region=region
        )
        if df.empty:
            return "查询成功，但未返回任何数据。请检查参数是否正确或该条件下是否有数据。"
        if target_label and target_time:
            try:
                clean_label = target_label.strip()
                clean_time = target_time.strip()
                value = df.loc[clean_label, clean_time]
                final_value = value.item() if hasattr(value, 'item') else value
                result_json = {
                    "source": "国家统计局(NBS)",
                    "kind": kind,
                    "path": path,
                    "query_params": {
                        "region": region,
                        "indicator": indicator,
                        "period": period
                    },
                    "result": {
                        "row_label": clean_label,
                        "column_label_time": clean_time,
                        "value": final_value
                    }
                }
                return result_json
            except KeyError:
                available_rows = df.index.to_list()
                available_cols = df.columns.to_list()
                return (
                    f"错误：找不到指定的行标签 '{target_label}' 或列标签 '{target_time}'。\n"
                    f"可用的行标签: {available_rows}\n"
                    f"可用的列标签: {available_cols}"
                )
        else:
            summary_json = {
                "summary": "数据查询成功，已获取到数据概览。",
                "guidance": "请提供 'target_label' (行标签) 和 'target_time' (列标签/时间) 以获取具体数值。",
                "data_preview": {
                    "available_row_labels_sample": available_rows[:10], 
                    "available_column_labels_sample": available_cols[:10] 
                }
            }
            return summary_json
    except Exception as e:
        return f"查询数据时发生错误: {e}。请检查您的参数是否与官网完全匹配。"

_financial_abstract_cache: Dict[str, pd.DataFrame] = {}

def get_net_profit_attributable(
    date: str,
    name: Optional[str] = None,
    code: Optional[str] = None
) -> Optional[float]:
    """
    查询指定公司在特定报告期的【归母净利润】（单位：元）。
    数据来源于新浪财经-财务报表-关键指标。

    Args:
        date (str): 要查询的具体报告期，格式应为 'YYYY-MM-DD'，例如 '2022-09-30'。
        name (Optional[str]): 股票的中文名称，例如 '白云机场'。
        code (Optional[str]): 股票的6位数字代码，例如 '600004'。

    Returns:
        Optional[float]: 查询到的归母净利润金额（元）。如果查询失败则返回 None。
    """
    global _financial_abstract_cache
    if not code and not name:
        print("错误：必须提供股票代码 (code) 或股票名称 (name)。")
        return None
    symbol_with_prefix = code if code else get_code_from_name(name)
    if not symbol_with_prefix:
        print(f"错误：未能通过名称 '{name}' 找到对应的股票代码。")
        return None
    symbol = re.sub(r'^[a-zA-Z]+', '', symbol_with_prefix)
    try:
        if symbol not in _financial_abstract_cache:
            print(f"缓存未命中，为代码'{symbol}'下载关键指标全量数据...")
            abstract_df = ak.stock_financial_abstract(symbol=symbol)
            if abstract_df is None or abstract_df.empty:
                print(f"错误：未能获取到代码 '{symbol}' 的关键指标数据。")
                _financial_abstract_cache[symbol] = pd.DataFrame()
                return None
            _financial_abstract_cache[symbol] = abstract_df
            print("数据下载并缓存成功。")
        df = _financial_abstract_cache[symbol]
        if df.empty:
            return None
        column_date = date.replace('-', '')
        if column_date not in df.columns:
            return f"查询失败：数据源中不存在报告期为 '{date}' ({column_date}) 的数据列。"
        profit_row = df[df['指标'] == '归母净利润']
        if profit_row.empty:
            return "查询失败：在返回的数据中未找到 '归母净利润' 这一指标。"
        value = profit_row.iloc[0][column_date]
        if pd.isna(value):
            return f"数据缺失：'{symbol}' 在 '{date}' 的归母净利润数据为空。"
        result_json = {
            "stock_identifier": name or symbol_with_prefix,
            "report_date": date,
            "financial_statement": "Financial Abstract",
            "item_name": "归母净利润",
            "value": float(value),
            "unit": "元"
        }
        return result_json
    except Exception as e:
        return f"处理数据时发生未知错误: {e}"

_benchmark_cache: Dict[str, pd.DataFrame] = {}

def get_sge_benchmark_price(
    metal: Literal["gold", "silver"],
    session: Literal["morning", "evening"],
    query_date: str = "latest"
) -> str:
    """
    查询上海黄金交易所(SGE)的【上海金】或【上海银】在特定日期的基准价。
    可指定查询早盘价(morning session)或晚盘价(evening session)。
    默认查询最新(latest)的交易日数据。
    """
    global _benchmark_cache
    api_map = {
        "gold": ak.spot_golden_benchmark_sge,
        "silver": ak.spot_silver_benchmark_sge
    }
    column_map = {
        "morning": "早盘价", # 午盘价，即早盘价
        "evening": "晚盘价"
    }
    metal_display_name = "上海金" if metal == "gold" else "上海银"
    session_display_name = "早盘(午盘)" if session == "morning" else "晚盘"
    try:
        if metal not in _benchmark_cache:
            print(f"--- [API 调用] 缓存未命中，正在通过 akshare 下载【{metal_display_name}】的全部历史基准价数据... ---")
            df = api_map[metal]()
            if df.empty:
                return f"错误: 从接口获取 '{metal_display_name}' 数据失败，返回为空。"
            df['交易时间'] = pd.to_datetime(df['交易时间'])
            df.set_index('交易时间', inplace=True)
            df.sort_index(ascending=True, inplace=True) 
            _benchmark_cache[metal] = df
        df = _benchmark_cache[metal]
        print(f"--- [函数缓存] 成功从缓存中读取【{metal_display_name}】数据。 ---")
    except Exception as e:
        return f"错误: 调用 akshare 接口或处理数据时失败: {e}"
    target_column = column_map[session]
    try:
        if query_date.lower() == "latest":
            latest_data = df.iloc[-1]
            price = latest_data[target_column]
            actual_date = latest_data.name.strftime('%Y-%m-%d')
        else:
            try:
                target_date = pd.to_datetime(query_date)
            except ValueError:
                return f"错误: 日期格式 '{query_date}' 无效。请使用 'YYYY-MM-DD' 格式或 'latest'。"
            closest_data = df.asof(target_date)
            if closest_data is None:
                min_date = df.index.min().strftime('%Y-%m-%d')
                return f"错误: 未能找到在 '{query_date}' 或此日期之前的任何有效数据。最早可用数据日期为 {min_date}。"
            price = closest_data[target_column]
            actual_date = closest_data.name.strftime('%Y-%m-%d')
        if pd.isna(price):
            return f"在 {actual_date} 找到了 {metal_display_name} 的记录，但其【{session_display_name}价】为空值。"
        result_json = {
            "source": "上海黄金交易所(SGE)",
            "benchmark_type": metal_display_name,
            "session": session_display_name,
            "date": actual_date,
            "price": float(price)
        }
        return result_json
    except KeyError:
        return f"错误: 数据源中缺少必需的列 '{target_column}'。"
    except Exception as e:
        return f"查询价格时发生未知错误: {e}"

_sge_report_cache: Optional[pd.DataFrame] = None

def get_sge_daily_data(product_name: str, query_date: str, column_label: str) -> str:
    """
    查询上海黄金交易所指定商品在特定日期的指定数据。
    能够处理'latest'等关键词，并自动查找最近的有效交易日数据。

    :param product_name: 要查询的商品名称 (例如 'Au(T+D)', 'Ag(T+D)')。
    :param query_date: 查询日期，格式为 'YYYY-MM-DD' 或 'latest' 等关键词。
    :param column_label: 要查询的数据列名 (例如 '收盘价', '成交量')。
    :return: 包含查询结果和数据日期的字符串，或详细的错误信息。
    """
    global _sge_report_cache
    try:
        if _sge_report_cache is None:
            print("--- [缓存未命中] 首次调用，正在下载完整的上海黄金交易所报告数据... ---")
            _sge_report_cache = ak.macro_china_au_report()
            _sge_report_cache['日期'] = _sge_report_cache['日期'].astype(str)
            print("--- [缓存成功] 数据已加载到内存中。 ---")
    except Exception as e:
        return f"错误：下载上海黄金交易所报告数据失败: {e}"
    df = _sge_report_cache
    available_products = df['商品'].unique()
    if product_name not in available_products:
        return f"错误：商品名称 '{product_name}' 无效。可用商品例如: 'Au(T+D)', 'Ag(T+D)', 'Au99.99' 等。"
    available_columns = df.columns.tolist()
    if column_label not in available_columns:
        return f"错误：列名 '{column_label}' 无效。可用列包括: {available_columns}。"
    effective_query_date_str = query_date
    LATEST_KEYWORDS = ['最新', 'latest', 'newest', 'today', '今天', '当前']
    if query_date.lower() in LATEST_KEYWORDS:
        print(f"--- [关键词识别] 检测到查询日期为 '{query_date}'，将查找最新交易日数据。---")
        effective_query_date_str = datetime.now().strftime('%Y-%m-%d')
    for i in range(7):
        try:
            current_date = datetime.strptime(effective_query_date_str, '%Y-%m-%d') - timedelta(days=i)
            current_date_str = current_date.strftime('%Y-%m-%d')
        except ValueError:
            return f"错误: 日期参数 '{query_date}' 格式无效。请使用 'YYYY-MM-DD' 格式或 'latest' 等关键词。"
        result_row = df[(df['日期'] == current_date_str) & (df['商品'] == product_name)]
        if not result_row.empty:
            try:
                value = result_row.iloc[0][column_label]
                result_json = {
                    "source": "上海黄金交易所(SGE)报告",
                    "product_name": product_name,
                    "date": current_date_str,
                    "metric_name": column_label,
                    "value": value
                }
                return result_json
            except KeyError:
                return f"错误：列名 '{column_label}' 无效。"
        else:
            if i == 0 and query_date.lower() in LATEST_KEYWORDS:
                print(f"--- 日期 '{current_date_str}' (今天) 无数据或非交易日，开始向前查找... ---")
            else:
                print(f"--- 日期 '{current_date_str}' 无数据，继续向前查找... ---")
    return f"错误: 在日期 '{effective_query_date_str}' 及其前7天内，均未找到商品 '{product_name}' 的有效交易数据。"

def get_stock_basic_info(
    symbol: str, 
    item_name: str, 
    report_date: Optional[str] = None, 
    us_report_type: str = "年报"
) -> Optional[Any]:
    """
    获取股票信息。
    - A股: 从东方财富获取基础信息。
    - 美股: 使用 AkShare 的【财务报表接口】获取详细行项目。
    
    :param symbol: 股票代码 (A股或美股)。
    :param item_name: 要查询的信息项。A股字段名为中文（如'公司名称'）；美股字段名为【报表项目中文名】（如'递延所得税资产'）。
    :param report_date: 指定报告日期 (可选)。
    :param us_report_type: 美股报告类型，可选 {"年报", "单季报", "累计季报"}，默认为 "年报"。
    :return: 查询到的指标值。
    """
    print(f"--- 正在为 '{symbol}' 查询基础信息: '{item_name}' ---")
    try:
        if re.match(r'^(SH|SZ|BJ)\d{6}$', symbol.upper()):
            market_code = symbol[2:]
            print(f"  -> 检测到 A 股代码，使用东方财富接口查询 '{market_code}'...")
            info_df = ak.stock_individual_info_em(symbol=market_code)
            if info_df is None or info_df.empty:
                return f"错误：未能获取到A股 '{symbol}' 的信息。"
            info_df.set_index("item", inplace=True)
            value = info_df.loc[item_name, 'value']
            result_json = {
                "stock_code": symbol,
                "stock_name": info_df.loc['公司名称', 'value'],
                "market": "A-Share",
                "data_source": "Basic Info",
                "requested_item": {
                    "name": item_name,
                    "value": value
                }
            }
            return result_json
        else:
            if us_report_type not in ["年报", "单季报", "累计季报"]:
                raise ValueError("美股报告类型 us_report_type 必须是 '年报', '单季报' 或 '累计季报' 之一。")
            report_name = "资产负债表" 
            print(f"  -> 按美股代码处理，使用【AkShare 财务报表接口】查询 '{report_name}' - '{us_report_type}' 的 '{item_name}'...")
            financial_report_df = ak.stock_financial_us_report_em(
                stock=symbol,
                symbol=report_name,
                indicator=us_report_type
            )
            if financial_report_df.empty:
                return f"错误：未能通过 AkShare 获取到 '{symbol}' 的 {report_name} 数据。"
            financial_report_df['REPORT_DATE'] = pd.to_datetime(financial_report_df['REPORT_DATE'])
            financial_report_df.sort_values(by='REPORT_DATE', ascending=False, inplace=True)
            filtered_df = financial_report_df[financial_report_df['ITEM_NAME'] == item_name].copy()
            if filtered_df.empty:
                available_items = financial_report_df['ITEM_NAME'].unique().tolist()
                print(f"错误：指标 '{item_name}' 不存在于当前报表 ({report_name}) 中。")
                print(f"当前报表可用项目（部分）：{available_items[:10]}")
                return None
            value = None
            actual_report_date = None
            if report_date:
                target_date = pd.to_datetime(report_date)
                filtered_df['date_diff'] = abs(filtered_df['REPORT_DATE'] - target_date)
                closest_report = filtered_df.loc[filtered_df['date_diff'].idxmin()]
                value = closest_report['AMOUNT']
                actual_report_date = closest_report['REPORT_DATE'].strftime('%Y-%m-%d')
            else:
                latest_report = filtered_df.iloc[0]
                value = latest_report['AMOUNT']
                actual_report_date = latest_report['REPORT_DATE'].strftime('%Y-%m-%d')
        if pd.notna(value):
            result_json = {
                    "stock_code": symbol,
                    "market": "US-Stock",
                    "data_source": f"Financial Report - {report_name} ({us_report_type})",
                    "report_date": actual_report_date,
                    "requested_item": {
                        "name": item_name,
                        "value": float(value),
                        "unit": "USD" 
                    }
                }
            return result_json
        else:
            print(f"*** 查询成功: '{item_name}' 的值是 -> 无数据 (NaN) ***")
            return None
    except Exception as e:
        print(f"获取或处理 '{symbol}' 数据时发生严重错误: {e}")
        return None


def get_stock_basic_info_a(symbol: str, item_name: str) -> Optional[Any]:
    """
    从东方财富获取指定A股股票的基础信息中的单个字段值。

    Args:
        symbol (str): 要查询的A股代码。支持两种格式: 
                    1. 纯6位数字代码, 如 '000001'。
                    2. 带交易所前缀的代码, 如 'SZ000001', 'SH600519'。
        item_name (str): 需要查询的信息项（字段名），必须是中文，例如 '公司名称', '总市值', '行业'。

    Returns:
        Optional[Any]: 查询到的具体数值，或在失败时返回 None。
    """
    print(f"--- 正在为 A股代码 '{symbol}' 查询基础信息: '{item_name}' ---")
    info_df = None
    try:
        if not re.search(r'(\d{6})', symbol):
            return f"错误：输入的 symbol '{symbol}' 格式不正确，无法提取6位数字代码。"
        market_code = re.search(r'(\d{6})', symbol).group(1)
        print(f"  -> 标准化代码为 '{market_code}'，使用东方财富接口查询...")
        info_df = ak.stock_individual_info_em(symbol=market_code)
        if info_df is None or info_df.empty:
            print(f"错误：未能获取到代码 '{market_code}' 的基础信息。请检查代码是否有效。")
            return None
        info_df.set_index("item", inplace=True)
        if item_name not in info_df.index:
            raise KeyError
        value = info_df.loc[item_name, 'value']
        stock_code_from_data = info_df.loc['代码', 'value']
        stock_name_from_data = info_df.loc['公司名称', 'value']
        result_json = {
            "stock_code": stock_code_from_data,
            "stock_name": stock_name_from_data,
            "market": "A-Share",
            "data_source": "Basic Info (Eastmoney)",
            "requested_item": {
                "name": item_name,
                "value": value
            }
        }
        return result_json
    except KeyError:
        print(f"错误：在 '{symbol}' 的信息中未找到名为 '{item_name}' 的字段。")
        if info_df is not None:
            print(f"可用字段包括: {info_df.index.tolist()}")
        return f"可用字段包括: {info_df.index.tolist()}"
    except Exception as e:
        return f"获取或处理 '{symbol}' 数据时发生严重错误: {e}"

_stock_comment_cache: Optional[pd.DataFrame] = None

def get_stock_comment(
    column_label: str,
    name: Optional[str] = None,
    code: Optional[str] = None
) -> Optional[Any]:
    global _stock_comment_cache
    symbol = code if code else get_code_from_name(name)
    if not symbol: return f"错误: 无法找到代码 for '{name or code}'."
    try:
        if _stock_comment_cache is None:
            print("首次查询，正在下载所有股票的'千股千评'数据...")
            df = ak.stock_comment_em()
            if df.empty: return "错误: 未能获取'千股千评'数据。"
            _stock_comment_cache = df
            print("数据缓存成功。")
        df = _stock_comment_cache
        stock_row = df[df['代码'] == symbol]
        if stock_row.empty:
            return f"查询失败: 在'千股千评'数据中未找到股票代码'{symbol}'。"
        value = float(stock_row.iloc[0][column_label])
        stock_name_from_data = stock_row.iloc[0].get("名称", name)
        result_json = {
            "stock_code": symbol,
            "stock_name": stock_name_from_data,
            "data_source": "千股千评 (Eastmoney)",
            "requested_item": {
                "name": column_label,
                "value": value
            }
        }
        return result_json
    except Exception as e:
        return f"查询'千股千评'时出错: {e}"
    
_ths_forecast_cache: Dict[tuple, Optional[pd.DataFrame]] = {}

def get_stock_consensus_forecast(
    metric: str,
    value_type: str,
    year: int,
    name: Optional[str] = None,
    code: Optional[str] = None,
) -> Any:
    """
    从同花顺查询指定A股在特定年度的【盈利预测】汇总数据。
    支持通过股票名称或代码查询，并使用缓存。
    """
    global _ths_forecast_cache
    if not code and not name: return "错误: 必须提供股票代码或名称。"
    symbol_from_input = code if code else get_code_from_name(name)
    if not symbol_from_input: return f"错误: 无法找到代码 for '{name or code}'."
    symbol = re.sub(r'\D', '', str(symbol_from_input))
    indicator_map = {"每股收益": "预测年报每股收益", "净利润": "预测年报净利润"}
    if metric not in indicator_map: return f"错误: 指标'{metric}'不受支持。"
    indicator = indicator_map[metric]
    try:
        cache_key = (symbol, indicator)
        if cache_key not in _ths_forecast_cache:
            print(f"缓存未命中，为代码'{symbol}'下载'{metric}'预测数据...")
            df = ak.stock_profit_forecast_ths(symbol=symbol, indicator=indicator)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df['年度'] = pd.to_numeric(df['年度'], errors='coerce')
                df.set_index('年度', inplace=True)
                _ths_forecast_cache[cache_key] = df
            else: _ths_forecast_cache[cache_key] = None
            print("数据缓存成功。")
        df = _ths_forecast_cache[cache_key]
        if df is None: return f"错误: 未能获取代码'{symbol}'的'{metric}'预测数据。"
        if year not in df.index:
            raise KeyError(f"年份 '{year}'")
        if value_type not in df.columns:
            raise KeyError(f"数值类型 '{value_type}'")
        result = df.loc[year, value_type]
        unit = ""
        final_value = result
        if metric == "净利润":
            final_value = result * 100000000 
            unit = "元"
        elif metric == "每股收益":
            unit = "元/股"
        result_json = {
            "stock_identifier": name or symbol_from_input,
            "forecast_year": year,
            "data_source": "盈利预测 (同花顺)",
            "requested_metric": {
                "metric_type": metric,
                "value_type": value_type,
                "value": final_value,
                "unit": unit
            }
        }
        return result_json
    except KeyError: return f"查询失败: 未找到'{year}'年度的预测记录。"
    except Exception as e: return f"查询盈利预测时出错: {e}"

_ths_forecast_cache: Dict[tuple, Optional[pd.DataFrame]] = {}
_earnings_announcement_cache: Dict[str, Optional[pd.DataFrame]] = {}

def _parse_chinese_number_unit(value_str: str) -> Optional[float]:
    if not isinstance(value_str, str): return value_str
    try:
        value_str = value_str.strip()
        if '亿' in value_str: return float(value_str.replace('亿', '')) * 1e8
        elif '万' in value_str: return float(value_str.replace('万', '')) * 1e4
        elif '%' in value_str: return float(value_str.replace('%', '')) / 100
        else: return float(value_str)
    except (ValueError, TypeError): return None
    
def get_stock_detailed_metric_forecast(
    metric_name: str,
    year: int,
    name: Optional[str] = None,
    code: Optional[str] = None,
) -> Any:
    """
    从同花顺查询A股未来特定年份的【详细财务指标预测】平均值。
    """
    global _ths_forecast_cache
    if not code and not name: return "错误: 必须提供股票代码或名称。"
    symbol_from_input = code if code else get_code_from_name(name)
    if not symbol_from_input: return f"错误: 无法找到代码 for '{name or code}'."
    symbol = re.sub(r'\D', '', str(symbol_from_input))
    indicator = "业绩预测详表-详细指标预测"
    try:
        cache_key = (symbol, indicator)
        if cache_key not in _ths_forecast_cache:
            print(f"缓存未命中，为代码'{symbol}'下载详细指标预测数据...")
            df = ak.stock_profit_forecast_ths(symbol=symbol, indicator=indicator)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.set_index('预测指标', inplace=True)
                _ths_forecast_cache[cache_key] = df
            else: 
                _ths_forecast_cache[cache_key] = None
            print("数据缓存成功。")
        df = _ths_forecast_cache[cache_key]
        if df is None: return f"错误: 未能获取代码'{symbol}'的详细指标预测数据。"
        column_name = f"预测{year}-平均"
        if metric_name not in df.index or column_name not in df.columns:
            return f"查询失败: 未找到指标'{metric_name}'或年份'{year}'的预测记录。"
        raw_value = df.loc[metric_name, column_name]
        final_value, unit = _parse_chinese_number_unit(raw_value)
        result_json = {
            "stock_identifier": name or symbol_from_input,
            "forecast_year": year,
            "data_source": "详细财务指标预测 (同花顺)",
            "requested_metric": {
                "name": metric_name,
                "value_type": "平均值",
                "value": final_value,
                "unit": unit
            }
        }
        return result_json
    except KeyError: return f"查询失败: 未找到指标'{metric_name}'或年份'{year}'的预测记录。"
    except Exception as e: return f"查询详细指标预测时出错: {e}"

_stock_zh_a_hist_cache: Dict[Tuple, Optional[pd.DataFrame]] = {}

def get_stock_hist_price_data(
    query_date: str,
    period: str,
    start_date: str,
    end_date: str,
    adjust: str,
    code: Optional[str] = None,
    name: Optional[str] = None
) -> Union[Dict[str, Any], str]: # <<< [修改] 更新了返回类型
    symbol = code if code else get_code_from_name(name)
    if not symbol:
        error_message = "错误: [get_stock_hist_price_data] 必须提供 'code' 或 'name'。"
        print(error_message)
        return error_message 
    cache_key = (symbol, period, start_date, end_date, adjust)
    if cache_key not in _stock_zh_a_hist_cache:
        try:
            print(f"新查询(东方财富)，正在下载股票 {symbol} 的行情数据...")
            df = ak.stock_zh_a_hist(
                symbol=symbol, period=period, start_date=start_date,
                end_date=end_date, adjust=adjust
            )
            if df is not None and not df.empty:
                df.set_index('日期', inplace=True)
                _stock_zh_a_hist_cache[cache_key] = df
            else:
                _stock_zh_a_hist_cache[cache_key] = None
        except Exception as e:
            error_message = f"错误: [get_stock_hist_price_data] 从东方财富下载 {symbol} 行情数据时出错: {e}"
            print(error_message)
            _stock_zh_a_hist_cache[cache_key] = None
            return error_message 
    df = _stock_zh_a_hist_cache[cache_key]
    if df is None or df.empty:
        error_message = f"错误: [get_stock_hist_price_data] 未能获取到 {symbol} 的数据（API返回为空或缓存失败）。"
        print(error_message)
        return error_message 
    try:
        data_series = df.loc[query_date]
        daily_data_dict = data_series.to_dict()
        result_json = {
            "stock_identifier": name or symbol,
            "query_date": query_date,
            "adjust_type": adjust if adjust else "non-adjusted",
            "period": period,
            "daily_quote": daily_data_dict 
        }
        return result_json
    except KeyError:
        error_message = f"错误: [get_stock_hist_price_data] 无法在 {symbol} 的数据中找到日期 '{query_date}'。请确认该日期是交易日且格式为 'YYYY-MM-DD'。"
        print(error_message)
        return error_message 
    except Exception as e:
        error_message = f"错误: [get_stock_hist_price_data] 处理 {symbol} 的行情数据时发生未知错误: {e}"
        print(error_message)
        return error_message 

_stock_zh_a_hist_tx_cache: Dict[Tuple, Optional[pd.DataFrame]] = {}

def get_stock_hist_price_data_tx(
    query_date: str,
    start_date: str,
    end_date: str,
    adjust: str,
    code: Optional[str] = None,
    name: Optional[str] = None
) -> Union[Dict[str, Any], str]: 
    symbol = code if code else get_code_from_name(name)
    if not symbol:
        error_message = "错误: [get_stock_hist_price_data_tx] 必须提供 'code' 或 'name'。"
        print(error_message)
        return error_message 
    cache_key = (symbol, start_date, end_date, adjust)
    if cache_key not in _stock_zh_a_hist_tx_cache:
        try:
            print(f"新查询(腾讯)，正在下载股票 {symbol} 从 {start_date} 到 {end_date} 的行情数据...")
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            if df is not None and not df.empty:
                df.set_index('date', inplace=True)
                _stock_zh_a_hist_tx_cache[cache_key] = df
            else:
                _stock_zh_a_hist_tx_cache[cache_key] = None
        except Exception as e:
            error_message = f"错误: [get_stock_hist_price_data_tx] 从腾讯证券下载 {symbol} 行情数据时出错: {e}"
            print(error_message)
            _stock_zh_a_hist_tx_cache[cache_key] = None
            return error_message 
    df = _stock_zh_a_hist_tx_cache[cache_key]
    if df is None or df.empty:
        error_message = f"错误: [get_stock_hist_price_data_tx] 未能获取到 {symbol} 的数据（API返回为空或缓存失败）。"
        print(error_message)
        return error_message
    try:
        data_series = df.loc[query_date]
        daily_data_dict = data_series.to_dict()
        result_json = {
            "stock_identifier": name or symbol,
            "data_source": "Tencent Finance",
            "query_date": query_date,
            "adjust_type": adjust if adjust else "non-adjusted",
            "daily_quote": daily_data_dict 
        }
        return result_json
    except KeyError:
        error_message = f"错误: [get_stock_hist_price_data_tx] 无法在 {symbol} 的数据中找到日期 '{query_date}'。请确认该日期是交易日且格式为 'YYYY-MM-DD'。"
        print(error_message)
        return error_message 
    except Exception as e:
        error_message = f"错误: [get_stock_hist_price_data_tx] 处理 {symbol} 的行情数据时发生未知错误: {e}"
        print(error_message)
        return error_message 

def get_stock_individual_basic_info_xq_value(row_label: str, name: Optional[str] = None, code: Optional[str] = None):
    symbol = None
    if code:
        symbol = code
    elif name:
        symbol = get_code_from_name(name)
    else:
        raise ValueError("必须提供 stock code ('code') 或 stock name ('name') 中的一个。")
    if not symbol:
        print(f"错误: 无法找到与输入相关的有效股票代码。")
        return None
    if symbol.startswith('6'):
        symbol = f"SH{symbol}"
    elif symbol.startswith(('0', '3')):
        symbol = f"SZ{symbol}"
    elif symbol.startswith(('4', '8')):
        symbol = f"BJ{symbol}"
    else:
        symbol = symbol 
    try:
        df = ak.stock_individual_basic_info_xq(symbol=symbol)
        df.set_index('item', inplace=True)
        if row_label not in df.index:
            raise KeyError
        value = df.loc[row_label, 'value']
        stock_code_from_data = df.loc['代码', 'value']
        stock_name_from_data = df.loc['名称', 'value']
        result_json = {
            "stock_code": stock_code_from_data,
            "stock_name": stock_name_from_data,
            "data_source": "雪球(Xueqiu) - Basic Info",
            "requested_item": {
                "name": row_label,
                "value": value
            }
        }
        return result_json
    except KeyError:
        print(f"错误: 无法在股票 '{symbol}' 的信息中找到项目 '{row_label}'。")
        if 'df' in locals():
            print(f"该股票可查询的项目有: {df.index.tolist()}")
        return None
    except Exception as e:
        print(f"获取股票 '{symbol}' 的数据时发生未知错误: {e}")
        return None

_long_stock_financial_cache = {}

def get_stock_long_short_term_investment(
    symbol: str, 
    query_date: str,
    investment_type: str = "long",  
    name: Optional[str] = None
) -> str:
    """
    查询A股公司在指定报告期的【长期股票投资】或【短期股票投资】金额（单位：元）。
    数据来源于新浪财经-财务指标接口（ak.stock_financial_analysis_indicator）。
    
    :param symbol: 股票代码（6位数字），例如 '600048'。
    :param query_date: 要查询的具体报告期，格式 'YYYY-MM-DD'。函数将返回该报告期的值。
    :param investment_type: 查询的投资类型。必须是 "long"（长期）或 "short"（短期），默认为 "long"。
    :param name: [可选] 股票的中文名称。
    :return: 返回查询到的具体数值（元）或错误信息。
    """
    global _long_stock_financial_cache
    if not symbol and name:
        return f"错误：请直接提供股票代码，或确保 get_code_from_name 函数可用。"
    if not symbol:
        return f"错误：未能找到股票 '{name}' 的A股代码。"
    if investment_type.lower() == "long":
        TARGET_COLUMN = "长期股票投资(元)"
    elif investment_type.lower() == "short":
        TARGET_COLUMN = "短期股票投资(元)"
    else:
        return f"错误：投资类型 '{investment_type}' 无效，请使用 'long' 或 'short'。"
    try:
        query_date_dt = pd.to_datetime(query_date)
        start_year = str(query_date_dt.year - 5) 
    except ValueError:
        return f"错误：日期格式 '{query_date}' 无效，请使用 'YYYY-MM-DD' 格式。"
    cache_key = (symbol, start_year)
    try:
        if cache_key in _long_stock_financial_cache:
            df = _long_stock_financial_cache[cache_key]
            print(f"缓存命中，使用股票 {symbol} 从 {start_year} 以来已缓存的财务指标数据。")
        else:
            print(f"缓存未命中，正在下载股票 {symbol} 从 {start_year} 以来的财务指标历史数据...")
            df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
            if df is None or df.empty or '日期' not in df.columns:
                return f"错误：未能获取股票 {symbol} 从 {start_year} 以来的财务指标数据。"
            _long_stock_financial_cache[cache_key] = df
            print("数据缓存成功。")
        df['日期'] = pd.to_datetime(df['日期'])
        if TARGET_COLUMN not in df.columns:
            return f"错误：接口返回的数据中不包含指标 '{TARGET_COLUMN}'。请检查接口文档。"
        target_row = df[df['日期'] == query_date_dt]
        if target_row.empty: 
            return f"查询失败: 股票 {symbol} 在报告期 {query_date} 未找到 {TARGET_COLUMN} 的数据。"
        value = target_row.iloc[0][TARGET_COLUMN]
        if pd.isna(value):
            return f"股票 {symbol} 在 {query_date} 报告期的 {TARGET_COLUMN} 值为: 无数据 (NaN)。"
        result_json = {
            "stock_identifier": name or symbol,
            "report_date": query_date,
            "data_source": "Financial Analysis Indicators",
            "item_name": TARGET_COLUMN,
            "value": float(value),
            "unit": "元"
        }
        return result_json
    except Exception as e:
        return f"查询 {TARGET_COLUMN} 数据时出错: {e}"

def get_stock_sse_deal_daily_value(date: str, row_label: str, column_label: str):
    """
    查询上交所(SSE)在特定交易日的单日成交概况。
    """
    try:
        formatted_date = pd.to_datetime(date).strftime('%Y%m%d')
        stock_sse_deal_daily_df = ak.stock_sse_deal_daily(date=formatted_date)
        if stock_sse_deal_daily_df is None or stock_sse_deal_daily_df.empty:
            print(f"警告: 日期 {date} 没有返回任何数据，可能是非交易日。")
            return None
        stock_sse_deal_daily_df.set_index('单日情况', inplace=True)
        if row_label not in stock_sse_deal_daily_df.index:
            raise KeyError(f"项目(行) '{row_label}' 不存在。")
        if column_label not in stock_sse_deal_daily_df.columns:
            raise KeyError(f"市场(列) '{column_label}' 不存在。")
        value = stock_sse_deal_daily_df.loc[row_label, column_label]
        result_json = {
            "market": "上海证券交易所(SSE)",
            "data_type": "Daily Deal Summary",
            "date": date,
            "requested_item": {
                "row": row_label,
                "column": column_label,
                "value": value
            }
        }
        return result_json
    except (KeyError, ValueError) as e:
        print(f"查询失败: {e}")
        if 'stock_sse_deal_daily_df' in locals() and stock_sse_deal_daily_df is not None:
            available_rows = stock_sse_deal_daily_df.index.tolist()
            available_cols = stock_sse_deal_daily_df.columns.tolist()
            print(f"可用项目(行)包括: {available_rows}")
            print(f"可用市场(列)包括: {available_cols}")
        return None
    except Exception as e:
        print(f"获取上交所单日成交概况时发生未知错误: {e}")
        return None

_top_10_sh_cache: Dict[tuple, pd.DataFrame] = {}
def get_top_10_shareholders(date: str, stock_code: str) -> Optional[str]:
    global _top_10_sh_cache
    symbol = re.sub(r'\D', '', str(stock_code))
    formatted_date = date.replace('-', '')
    if re.match(r'^6', symbol): prefixed_symbol = f"sh{symbol}"
    else: prefixed_symbol = f"sz{symbol}"
    cache_key = (prefixed_symbol, formatted_date)
    try:
        if cache_key not in _top_10_sh_cache:
            print(f"缓存未命中，为代码'{prefixed_symbol}'在'{formatted_date}'下载十大股东数据...")
            df = ak.stock_gdfx_top_10_em(symbol=prefixed_symbol, date=formatted_date)
            _top_10_sh_cache[cache_key] = df if isinstance(df, pd.DataFrame) else None
            print("数据缓存成功。")
        df = _top_10_sh_cache[cache_key]
        if df is None: return f"错误: 未能获取数据。可能日期'{date}'无效或非报告期。"
        shareholders_list = []
        target_columns = ['名次', '股东名称', '持股数', '占总股本持股比例', '增减']
        if not all(col in df.columns for col in target_columns):
            return f"错误: 返回的数据缺少必要的列。需要: {target_columns}"
        filtered_df = df[target_columns]
        for index, row in filtered_df.iterrows():
            shareholder_info = {
                "rank": row.get('名次'),
                "name": row.get('股东名称'),
                "shares_held": row.get('持股数'),
                "percentage_of_total_shares": row.get('占总股本持股比例'),
                "change_status": row.get('增减')
            }
            shareholders_list.append(shareholder_info)
        result_json = {
            "stock_identifier": stock_code,
            "report_date": date,
            "data_source": "Top 10 Shareholders (Eastmoney)",
            "shareholders": shareholders_list
        }
        return result_json
    except ValueError as ve:
        if "Length mismatch" in str(ve):
            _top_10_sh_cache[cache_key] = None
            return f"错误: 日期'{date}'很可能不是一个有效的报告期末，导致底层库查询失败。"
        else:
            return f"查询十大股东时发生值错误: {ve}"
    except Exception as e:
        return f"查询十大股东时发生未知错误: {e}"

def calculate_price_change_pct_in_period(
    start_date: str,
    end_date: str,
    market: str,
    adjust: str = 'qfq',
    name: Optional[str] = None,
    code: Optional[str] = None
) -> str:
    """
    计算单只股票在【指定时间段内】的累计涨跌幅（百分比）。
    """
    if not code and not name:
        return "错误：必须提供股票代码 (code) 或股票名称 (name)。"
    if adjust not in ['', 'qfq', 'hfq']:
        return f"错误: 'adjust' 参数 '{adjust}' 无效。有效选项: '', 'qfq', 'hfq'。"
    try:
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return f"错误: 日期格式无效。请确保 start_date 和 end_date 均为 'YYYY-MM-DD' 格式。"
    if start_date_dt >= end_date_dt:
        return f"错误: 开始日期 {start_date} 必须早于结束日期 {end_date}。"
    effective_code = code
    effective_name = name
    if not effective_code and name:
        print(f"--- [代码解析] 缺少代码，正在尝试通过名称 '{name}' (市场: {market}) 查找代码... ---")
        effective_code = get_code_from_name(name, market=market)
        if not effective_code:
            return f"错误：通过名称 '{name}' 在市场 '{market}' 未能找到有效的股票代码。"
        print(f"--- [代码解析] 成功找到代码: {effective_code} ---")
    if not effective_code:
        return "错误：最终未能获得一个有效的股票代码用于查询。"
    history_fetcher_map = {
        'a': get_a_stock_daily_price,
        'hk': get_hk_stock_daily_price,
        'us': get_us_stock_daily_price
    }
    if market not in history_fetcher_map:
        return f"错误：无效的市场类型 '{market}'。支持的市场: 'a', 'hk', 'us'。"
    history_fetcher = history_fetcher_map[market]
    print(f"--- 正在获取代码 {effective_code} 从 {start_date} 到 {end_date} 的历史收盘价... ---")
    try:
        hist_data_str = history_fetcher(
            code=effective_code,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        hist_data_list = json.loads(hist_data_str)
        if not isinstance(hist_data_list, list) or not hist_data_list:
            return f"错误: 在指定时间段内未能获取到代码'{effective_code}'的任何交易数据。底层工具返回: {hist_data_str}"
        first_day_data = hist_data_list[0]
        last_day_data = hist_data_list[-1]
        start_price = float(first_day_data['close'])
        end_price = float(last_day_data['close'])
        actual_start_date = first_day_data['date']
        actual_end_date = last_day_data['date']
    except (json.JSONDecodeError, TypeError):
        return f"错误：解析代码'{effective_code}'的历史数据时失败。底层工具返回的不是有效的JSON列表: {hist_data_str}"
    except (IndexError, KeyError, ValueError) as e:
        return f"错误：返回的历史数据格式不正确，无法提取首末日期或收盘价。底层工具返回: {hist_data_str} (错误: {e})"
    except Exception as e:
        return f"错误：获取代码'{effective_code}'的历史数据时发生未知错误: {e}。底层工具返回: {hist_data_str}"
    try:
        if start_price == 0:
            return f"错误: 计算错误，起始交易日 ({actual_start_date}) 的收盘价为0，无法计算涨跌幅。"
        change_pct = ((end_price - start_price) / start_price) * 100
        result_json = {
            "analysis_type": "stock_change_percentage_in_period",
            "stock_identifier": effective_name or effective_code, 
            "stock_code": effective_code,
            "market": market,
            "query_period": {
                "start": start_date,
                "end": end_date
            },
            "actual_trading_period": {
                "start_date": actual_start_date,
                "end_date": actual_end_date
            },
            "period_prices": {
                "start_price": f"{start_price:.2f}",
                "end_price": f"{end_price:.2f}"
            },
            "calculation_result": {
                "cumulative_percentage_change": f"{change_pct:+.2f}%"
            }
        }
        return result_json
    except Exception as e:
        return f"错误：在计算累计涨跌幅时发生未知异常: {e}"

def get_stocks_history_dataframe(
    codes: List[str], 
    start_date: str, 
    end_date: str, 
    adjust: str = ""
) -> Union[pd.DataFrame, str]: 
    """
    一次性获取【多只股票】在一段时间内的历史行情数据。
    该函数通过在日期范围内循环调用单点查询工具 (如 get_a_stock_daily_price) 来构建 DataFrame。

    Args:
        codes (List[str]): 股票代码的列表 (例如: ["sh600519", "sz300750"]).
        start_date (str): 开始日期, 格式 'YYYY-MM-DD'.
        end_date (str): 结束日期, 格式 'YYYY-MM-DD'.
        adjust (str, optional): 复权类型. "", "qfq", "hfq". 默认为 "".

    Returns:
        Union[pd.DataFrame, str]: 
            - 成功: 一个包含所有股票历史数据的、合并好的 DataFrame。
            - 失败: 一个包含错误信息的字符串。
    """
    if not isinstance(codes, list) or not codes:
        error_message = "错误：'codes' 参数必须是一个非空的列表，例如 ['sh600519']。"
        print(f"--- [批量获取DataFrame v2] {error_message} ---")
        return error_message
    VALID_ADJUSTS = ["", "qfq", "hfq"]
    if adjust not in VALID_ADJUSTS:
        error_message = f"错误：无效的 'adjust' 参数 '{adjust}'。有效选项: {VALID_ADJUSTS}"
        print(f"--- [批量获取DataFrame v2] {error_message} ---")
        return error_message
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    except ValueError as e:
        error_message = f"错误：日期格式不正确或范围无效 (start: '{start_date}', end: '{end_date}')。请使用 'YYYY-MM-DD' 格式。错误: {e}"
        print(f"--- [批量获取DataFrame v2] {error_message} ---")
        return error_message
    all_daily_records = []
    market_dispatch = {}
    
    # --- 修改开始 ---
    # 优先规范化股票代码，确保A股代码有前缀
    processed_codes = []
    for code in codes:
        if code.isdigit() and len(code) == 6: # 假设6位数字是中国A股
            if code.startswith(('60', '00', '30', '68')): # 根据A股代码常见开头判断
                # 尝试自动添加前缀
                if code.startswith('6'): # 沪市A股
                    processed_codes.append(f"sh{code}")
                elif code.startswith(('00', '30', '68')): # 深市A股 (00, 30, 68)
                    processed_codes.append(f"sz{code}")
                else: # 无法确定市场，保留原始，让下面的dispatch处理
                    processed_codes.append(code)
            else: # 纯数字但不是A股的常见开头，保留原始
                processed_codes.append(code)
        else: # 非6位数字的纯数字或已带前缀的，保留原始
            processed_codes.append(code)

    # 重新构建 market_dispatch
    for code in processed_codes: # 使用处理后的代码列表
        if code.startswith('sh') or code.startswith('sz') or code.startswith('bj'):
            market_dispatch[code] = get_a_stock_daily_price
        # 注意：此处要确保纯数字的A股代码已经被上面的逻辑处理成带前缀的了
        # 否则，如果'300750'进来，它会走到get_hk_stock_daily_price (如果 len <= 5 是错的)
        # 或者 get_us_stock_daily_price (如果上面A股识别不够全面)
        elif code.isdigit() and (len(code) == 4 or len(code) == 5): # 港股通常4-5位数字
            market_dispatch[code] = get_hk_stock_daily_price
        else: # 假设其他都是美股或未能识别
            market_dispatch[code] = get_us_stock_daily_price
            
    print(f"--- [批量获取DataFrame v2] 将为 {len(processed_codes)} 只股票查询 {len(date_range)} 天的数据... ---")
    
    # 修改循环，使用 processed_codes
    for code in processed_codes:
    # --- 修改结束 ---
        price_fetcher = market_dispatch.get(code)
        if not price_fetcher:
            print(f"    [警告] 无法为代码 '{code}' 确定市场类型，已跳过。")
            continue
        # ... (其余代码不变) ...
        columns_to_fetch = ['open', 'high', 'low', 'close', 'amount'] 
        for current_date in date_range:
            date_str = current_date.strftime('%Y-%m-%d')
            daily_data = {'code': code, 'date': date_str}
            is_successful_day = True
            for col in columns_to_fetch:
                try:
                    result_dict = price_fetcher(
                        code=code, 
                        query_date=date_str, 
                        column_label=col, 
                        adjust=adjust
                    )
                    if isinstance(result_dict, dict) and 'requested_item' in result_dict:
                        if result_dict.get('date') == date_str:
                            daily_data[col] = result_dict['requested_item']['value']
                        else:
                            is_successful_day = False
                            break 
                    else:
                        is_successful_day = False
                        break
                except Exception as e:
                    print(f"    [警告] 在为 {code} 获取 {date_str} 的 '{col}' 数据时发生内部错误: {e}")
                    is_successful_day = False
                    break
            if is_successful_day:
                all_daily_records.append(daily_data)
    if not all_daily_records:
        error_message = f"错误：在为股票 {codes} 查询 {start_date} 到 {end_date} 期间的数据时，未能收集到任何有效的交易日记录。"
        print(f"--- [批量获取DataFrame v2] {error_message} ---")
        return error_message 
    final_df = pd.DataFrame(all_daily_records)
    final_df['date'] = pd.to_datetime(final_df['date'])
    if 'amount' in final_df.columns:
        final_df.rename(columns={'amount': 'volume'}, inplace=True)
    print(f"--- [批量获取DataFrame v2] 完成！成功构建了包含 {len(final_df)} 条记录的 DataFrame。 ---")
    return final_df

def python_interpreter(code: str) -> Any:
    """
    执行一段 Python 代码字符串并返回其最终表达式的结果。
    代码可以访问 pandas 库 (别名为 pd) 以及已在此环境中定义的其他变量。
    
    Args:
        code (str): 一段有效的 Python 代码字符串。为了返回值，代码的最后一行
                    必须是一个可以被 'eval()' 求值的表达式。

    Returns:
        Any: 代码最后一行表达式的执行结果。
    """
    print(f"--- [Python Interpreter] 正在执行以下代码 ---\n{code}\n---------------------------------------------")
    local_scope = {
        'pd': pd
    }
    try:
        lines = code.strip().split('\n')
        if len(lines) > 1:
            exec('\n'.join(lines[:-1]), globals(), local_scope)
        result = eval(lines[-1], globals(), local_scope)
        print(f"--- [Python Interpreter] 执行成功，返回类型: {type(result)} ---")
        return result
    except Exception as e:
        import traceback
        error_message = f"Python代码执行失败: {e}\n{traceback.format_exc()}"
        print(f"--- [Python Interpreter] {error_message} ---")
        return {"error": error_message}

def get_stocks_gain_ranking_dataframe(
    codes: List[str],
    start_date: str,
    end_date: str,
    adjust: str = ""
) -> pd.DataFrame:
    """
    【批量分析工具】计算【多只股票】在指定时间段内的区间涨跌幅，并按涨幅从高到低进行排名。
    直接返回一个包含排名结果的 DataFrame。

    Args:
        codes (List[str]): 股票代码的列表 (例如: ["sh600519", "sz300750"]).
        start_date (str): 开始日期, 格式 'YYYY-MM-DD'.
        end_date (str): 结束日期, 格式 'YYYY-MM-DD'.
        adjust (str, optional): 复权类型. "", "qfq", "hfq". 默认为 "".

    Returns:
        pd.DataFrame: 一个包含排名结果的 DataFrame，列包括 'rank', 'code', 'name', 'gain_percentage'。
    """
    print(f"--- [批量排名分析] 正在为 {len(codes)} 只股票计算从 {start_date} 到 {end_date} 的涨跌幅排名... ---")
    history_df = get_stocks_history_dataframe(
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust
    )
    if history_df.empty:
        print("--- [批量排名分析] 未能获取到任何历史数据，返回空排名。 ---")
        return pd.DataFrame(columns=['rank', 'code', 'name', 'gain_percentage'])
    def calculate_gain(group):
        group = group.sort_values('date')
        start_price = group.iloc[0]['close']
        end_price = group.iloc[-1]['close']
        if start_price is None or pd.isna(start_price) or start_price == 0:
            return None 
        gain = ((end_price - start_price) / start_price) * 100
        return gain
    ranking_series = history_df.groupby('code').apply(calculate_gain)
    ranking_df = ranking_series.reset_index(name='gain_percentage')
    ranking_df.dropna(inplace=True)
    if ranking_df.empty:
        return pd.DataFrame(columns=['rank', 'code', 'name', 'gain_percentage'])
    ranking_df.sort_values('gain_percentage', ascending=False, inplace=True)
    ranking_df.reset_index(drop=True, inplace=True)
    ranking_df['rank'] = ranking_df.index + 1
    code_to_name_map = dict(zip(history_df['code'], history_df.get('name', history_df['code'])))
    ranking_df['name'] = ranking_df['code'].map(code_to_name_map)
    ranking_df['gain_percentage'] = ranking_df['gain_percentage'].round(2)
    final_columns = ['rank', 'code', 'name', 'gain_percentage']
    final_df = ranking_df[[col for col in final_columns if col in ranking_df.columns]]
    print(f"--- [批量排名分析] 完成！成功计算并排名了 {len(final_df)} 只股票。 ---")
    return final_df
    
def calculate_index_change_pct_in_period(
    identifier: str,
    start_date: str,
    end_date: str,
    market: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    计算单个指数在【指定时间段内】的累计涨跌幅（百分比）。
    函数会利用 get_index_history 自动查找此时间段内的第一个和最后一个实际交易日的收盘价进行计算。

    :param identifier: 指数名称或代码, 例如 "沪深300" 或 "纳斯达克"。
    :param start_date: 查询周期的开始日期 (格式 'YYYY-MM-DD')。
    :param end_date: 查询周期的结束日期 (格式 'YYYY-MM-DD')。
    :param market: 市场提示 (可选), 例如 'a', 'hk', 'us'。
    :return: 包含详细计算结果的字典，或描述错误的字符串。
    """
    print(f"--- [指数涨跌幅计算] 正在计算 '{identifier}' 从 {start_date} 到 {end_date} 的涨跌幅... ---")
    if not identifier:
        return "错误：必须提供指数的名称或代码 (identifier)。"
    try:
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return f"错误: 日期格式无效。请确保 start_date 和 end_date 均为 'YYYY-MM-DD' 格式。"
    if start_date_dt >= end_date_dt:
        return f"错误: 开始日期 {start_date} 必须早于结束日期 {end_date}。"
    print(f"--- 正在获取起始点 '{start_date}' 的数据... ---")
    start_data_result = get_index_history(
        identifier=identifier,
        query_date=start_date,
        column_label='close', 
        market=market
    )
    if isinstance(start_data_result, str):
        return f"错误：获取开始日数据时失败: {start_data_result}"
    try:
        start_price = float(start_data_result['value'])
        actual_start_date = start_data_result['date']
        standard_identifier = start_data_result['identifier'] 
    except (KeyError, TypeError) as e:
        return f"错误：解析开始日数据时返回的格式不正确: {start_data_result} (错误: {e})"
    print(f"--- 正在获取结束点 '{end_date}' 的数据... ---")
    end_data_result = get_index_history(
        identifier=identifier,
        query_date=end_date,
        column_label='close',
        market=market
    )
    if isinstance(end_data_result, str):
        return f"错误：获取结束日数据时失败: {end_data_result}"
    try:
        end_price = float(end_data_result['value'])
        actual_end_date = end_data_result['date']
    except (KeyError, TypeError) as e:
        return f"错误：解析结束日数据时返回的格式不正确: {end_data_result} (错误: {e})"
    try:
        if start_price == 0:
            return f"错误: 计算错误，起始交易日 ({actual_start_date}) 的收盘价为0，无法计算涨跌幅。"
        change_pct = ((end_price - start_price) / start_price) * 100
        result_dict = {
            "analysis_type": "index_change_percentage_in_period",
            "index_identifier": standard_identifier,
            "market_hint": market,
            "query_period": {
                "start": start_date,
                "end": end_date
            },
            "actual_trading_period": {
                "start_date": actual_start_date,
                "end_date": actual_end_date
            },
            "period_prices": {
                "start_price": f"{start_price:.2f}",
                "end_price": f"{end_price:.2f}"
            },
            "calculation_result": {
                "cumulative_percentage_change": f"{change_pct:+.2f}%"
            }
        }
        return result_dict
    except Exception as e:
        return f"错误：在计算累计涨跌幅时发生未知异常: {e}"

def _fetch_single_index_history_range(
    identifier: str, 
    start_date: str, 
    end_date: str
) -> Union[pd.DataFrame, str]:
    """
    获取单个指数在指定时间段内的全部历史数据DataFrame。
    """
    print(f"--- [数据获取子任务] 正在获取 '{identifier}' 从 {start_date} 到 {end_date} 的数据... ---")
    entity_info = find_index_code_and_market(identifier=identifier)
    if not entity_info: return f"实体链接失败: '{identifier}'"
    code_for_api, name_for_api, identified_market = entity_info
    market_for_api = identified_market.replace("_index", "")
    all_history_df = None
    api_sources = []
    start_date_fmt = start_date.replace('-', '')
    end_date_fmt = end_date.replace('-', '')
    if market_for_api == 'a':
        api_sources = [
            lambda: ak.index_zh_a_hist(symbol=code_for_api[2:], period="daily", start_date=start_date_fmt, end_date=end_date_fmt),
            lambda: ak.stock_zh_index_daily(symbol=code_for_api)
        ]
    elif market_for_api == 'hk':
        api_sources = [lambda: ak.stock_hk_index_daily_sina(symbol=code_for_api)]
    elif market_for_api == 'us':
        us_symbol = f".{code_for_api.upper().lstrip('.')}"
        api_sources = [lambda: ak.index_us_stock_sina(symbol=us_symbol)]
    elif market_for_api == 'global':
        api_sources = [lambda: ak.index_global_hist_em(symbol=name_for_api, start_date=start_date_fmt, end_date=end_date_fmt)]
        US_INDEX_IDENTIFIERS = ['IXIC', 'DJI', 'SPX', '纳斯达克', '道琼斯', '标普']
        if any(us_id in code_for_api.upper() or us_id in name_for_api for us_id in US_INDEX_IDENTIFIERS):
            us_symbol = f".{code_for_api.upper().lstrip('.')}"
            api_sources.insert(0, lambda: ak.index_us_stock_sina(symbol=us_symbol))
    for fetch_func in api_sources:
        try:
            temp_df = fetch_func()
            if temp_df is not None and not temp_df.empty:
                all_history_df = temp_df
                break
        except Exception as e:
            print(f"--- [数据获取子任务] 接口调用失败: {e} ---")
            continue
    if all_history_df is None or all_history_df.empty:
        return f"所有接口均未能获取到 '{identifier}' 的数据。"
    all_history_df.rename(columns=_COLUMN_MAP_AK_TO_STD, inplace=True, errors='ignore')
    if 'date' not in all_history_df.columns:
        return f"获取到的 '{identifier}' 数据缺少'date'列。"
    all_history_df['date'] = pd.to_datetime(all_history_df['date'])
    mask = (all_history_df['date'] >= start_date) & (all_history_df['date'] <= end_date)
    filtered_df = all_history_df.loc[mask].copy()
    filtered_df['identifier'] = name_for_api 
    return filtered_df


def get_multiple_index_history_df(
    identifiers: List[str],
    start_date: str,
    end_date: str,
    columns_to_include: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取【多个指数】在【指定时间段内】的详细历史行情数据，并以DataFrame对象返回。
    
    :param identifiers: 一个包含多个指数名称或代码的列表。
    :param start_date: 开始日期 (格式 'YYYY-MM-DD')。
    :param end_date: 结束日期 (格式 'YYYY-MM-DD')。
    :param columns_to_include: (可选) 一个包含所需数据列名的列表。
    :return: 一个包含历史数据的 pandas.DataFrame 对象。如果发生错误，则返回一个空的DataFrame。
    """
    print(f"--- [批量获取任务] 开始获取 {identifiers} 从 {start_date} 到 {end_date} 的历史数据... ---")
    VALID_COLUMNS = {'open', 'high', 'low', 'close', 'volume', 'turnover'}
    if columns_to_include:
        invalid_columns = [col for col in columns_to_include if col not in VALID_COLUMNS]
        if invalid_columns:
            error_message = f"错误：请求了无效的列名 {invalid_columns}。有效的列名包括: {list(VALID_COLUMNS)}。"
            print(f"--- [批量获取任务] {error_message}")
            return error_message
    else:
        columns_to_include = ['open', 'high', 'low', 'close', 'volume', 'turnover']
    all_dfs = []
    errors = []
    for identifier in identifiers:
        result = _fetch_single_index_history_range(identifier, start_date, end_date)
        if isinstance(result, pd.DataFrame):
            all_dfs.append(result)
        else:
            errors.append(f"'{identifier}': {result}")
    if not all_dfs:
        if not all_dfs:
            error_message = f"错误：未能成功获取任何一个指数的数据。详情: {'; '.join(errors)}"
            print(f"--- [批量获取任务] {error_message} ---")
            return error_message
    if errors:
        error_message = f"错误：部分指数获取失败: {'; '.join(errors)} ---"
        print(f"--- [批量获取任务] {error_message}")
        return error_message
    try:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        required_cols = ['date', 'identifier'] + columns_to_include
        final_cols_to_use = [col for col in required_cols if col in combined_df.columns]
        filtered_df = combined_df[final_cols_to_use]
        value_columns = [col for col in final_cols_to_use if col not in ['date', 'identifier']]
        if not value_columns:
            error_message = f"错误：请求的列 {columns_to_include} 在获取到的数据中均不存在。---"
            print(f"--- [批量获取任务] 警告：请求的列 {columns_to_include} 在获取到的数据中均不存在。---")
            return error_message
        pivot_df = filtered_df.pivot_table(
            index='date', 
            columns='identifier', 
            values=value_columns
        )
        return pivot_df
    except Exception as e:
        error_message = f"错误：在合并和处理数据时发生异常: {e} ---"
        print(f"--- [批量获取任务] 错误：在合并和处理数据时发生异常: {e} ---")
        return error_message

def get_indices_gain_ranking_dataframe(
    identifiers: List[str],
    start_date: str,
    end_date: str
) -> Union[pd.DataFrame, str]:
    """
    【批量分析工具】计算【多只指数】在指定时间段内的区间涨跌幅，并按涨幅从高到低进行排名。
    严格按照要求，在成功时直接返回一个 pandas.DataFrame 对象。

    :param identifiers: 指数名称或代码的列表 (例如: ["沪深300", "纳斯达克"])。
    :param start_date: 开始日期, 格式 'YYYY-MM-DD'.
    :param end_date: 结束日期, 格式 'YYYY-MM-DD'.
    :return: 一个包含排名结果的 pandas.DataFrame 对象，或描述错误的字符串。
    """
    print(f"--- [批量排名分析] 正在为 {len(identifiers)} 只指数计算从 {start_date} 到 {end_date} 的涨跌幅排名... ---")
    history_df_wide = get_multiple_index_history_df(
        identifiers=identifiers,
        start_date=start_date,
        end_date=end_date,
        columns_to_include=['close']
    )
    if isinstance(history_df_wide, str):
        return f"错误：在获取批量历史数据时失败: {history_df_wide}"
    if history_df_wide.empty:
        return "错误：未能获取到任何指数在指定时间段内的历史数据。"
    history_df_long = history_df_wide.stack().reset_index()
    history_df_long.columns = ['date', 'identifier', 'close']
    history_df_long['date'] = pd.to_datetime(history_df_long['date'])
    def calculate_gain(group):
        group = group.sort_values('date')
        if len(group) < 2: return None
        start_price = group.iloc[0]['close']
        end_price = group.iloc[-1]['close']
        if pd.isna(start_price) or start_price == 0: return None
        return ((end_price - start_price) / start_price) * 100
    ranking_series = history_df_long.groupby('identifier').apply(calculate_gain)
    ranking_df = ranking_series.reset_index(name='gain_percentage')
    ranking_df.dropna(inplace=True)
    if ranking_df.empty:
        return "信息：所有指数都因数据不足而无法计算涨跌幅。"
    ranking_df.sort_values('gain_percentage', ascending=False, inplace=True)
    ranking_df.reset_index(drop=True, inplace=True)
    ranking_df['rank'] = ranking_df.index + 1
    ranking_df['gain_percentage'] = ranking_df['gain_percentage'].round(2)
    final_df = ranking_df[['rank', 'identifier', 'gain_percentage']]
    print(f"--- [批量排名分析] 完成！成功计算并排名了 {len(final_df)} 只指数。 ---")
    return final_df

def analyze_index_performance(
    identifiers: List[str],
    start_date: str,
    end_date: str,
    pct_change_threshold_percent: Optional[float] = None,
    turnover_threshold_yuan: Optional[float] = None
) -> Union[Dict, str]:
    """
    分析一个或多个指数在指定时间段内的表现。
    此工具能获取原始数据，计算每日涨跌幅，并根据给定的阈值进行筛选，最终返回满足所有条件的天数和具体日期。
    
    :param identifiers: 指数名称或代码的列表。
    :param start_date: 开始日期, 格式 'YYYY-MM-DD'.
    :param end_date: 结束日期, 格式 'YYYY-MM-DD'.
    :param pct_change_threshold_percent: (可选) 日涨跌幅筛选阈值(%)。例如，输入1.0代表筛选出涨幅 > 1% 的天数。
    :param turnover_threshold_yuan: (可选) 日成交金额筛选阈值(元)。例如，输入5000e8代表筛选出成交金额 > 5000亿元的天数。
    :return: 一个包含分析结果的字典，或描述错误的字符串。
    """
    print(f"--- [高级分析任务] 开始分析 {identifiers} 从 {start_date} 到 {end_date} 的表现... ---")
    required_columns = ['close', 'turnover']
    history_df = get_multiple_index_history_df(
        identifiers=identifiers,
        start_date=start_date,
        end_date=end_date,
        columns_to_include=required_columns
    )
    if history_df.empty:
        return "错误：未能获取到用于分析的基础行情数据，无法继续。"
    results = {}
    for identifier in identifiers:
        if ('close', identifier) not in history_df.columns:
            results[identifier] = {"status": "error", "message": "缺少收盘价数据，无法分析。"}
            continue
        index_df = history_df[[col for col in history_df.columns if col[1] == identifier]].copy()
        index_df.columns = index_df.columns.droplevel(1)
        index_df['pct_change'] = index_df['close'].pct_change() * 100
        conditions = pd.Series(True, index=index_df.index) 
        if pct_change_threshold_percent is not None:
            conditions &= index_df['pct_change'] > pct_change_threshold_percent
        if turnover_threshold_yuan is not None:
            if 'turnover' in index_df.columns:
                conditions &= index_df['turnover'] > turnover_threshold_yuan
            else:
                results[identifier] = {"status": "error", "message": "数据源未提供成交金额(turnover)，无法按此条件筛选。"}
                continue
        filtered_days = index_df[conditions]
        results[identifier] = {
            "status": "success",
            "count_of_matching_days": len(filtered_days),
            "matching_days_details": [
                {
                    "date": day.strftime('%Y-%m-%d'),
                    "pct_change": round(row.pct_change, 2) if pd.notna(row.pct_change) else None,
                    "turnover_yuan": int(row.turnover) if 'turnover' in row and pd.notna(row.turnover) else None
                }
                for day, row in filtered_days.iterrows()
            ]
        }
    return results

def format_date(stat_time_obj):
    s = str(stat_time_obj)
    parts = s.split('.')
    if len(parts) == 2:
        year = parts[0]
        month = int(parts[1])
        return f"{year}-{month:02d}"
    return s

@lru_cache(maxsize=1)
def _get_and_clean_gold_forex_data() -> Union[pd.DataFrame, str]:
    df = pd.DataFrame()
    try:
        print("--- [数据接口] 正在尝试主数据源 (新浪财经)... ---")
        df_sina = ak.macro_china_foreign_exchange_gold()
        def format_date_sina(stat_time_obj):
            s = str(stat_time_obj)
            parts = s.split('.')
            if len(parts) == 2:
                year = parts[0]
                month = int(parts[1])
                return f"{year}-{month:02d}"
            return s
        df_sina['统计时间'] = df_sina['统计时间'].apply(format_date_sina)
        df = df_sina[['统计时间', '黄金储备', '国家外汇储备']]
        print("--- [数据接口] 主数据源 (新浪财经) 获取成功。 ---")
    except Exception as e_sina:
        print(f"--- [数据接口] 主数据源 (新浪财经) 获取失败: {e_sina} ---")
        print("--- [数据接口] 正在尝试切换到您指定的备用数据源 (东方财富)... ---")
        try:
            df_em = ak.macro_china_fx_gold()
            df_em = df_em[['月份', '黄金储备-数值', '国家外汇储备-数值']]
            df_em.rename(columns={
                '月份': '统计时间',
                '黄金储备-数值': '黄金储备',
                '国家外汇储备-数值': '国家外汇储备'
            }, inplace=True)
            df_em['统计时间'] = df_em['统计时间'].str.replace('年', '-').str.replace('月份', '')
            df = df_em
            print("--- [数据接口] 备用数据源 (东方财富) 获取成功。 ---")
        except Exception as e_em:
            error_message = f"错误：备用数据源 (东方财富) 也获取失败: {e_em} ---"
            print(f"--- [数据接口] 备用数据源 (东方财富) 也获取失败: {e_em} ---")
            return error_message
    if not df.empty:
        df.dropna(inplace=True)
        df.sort_values('统计时间', inplace=True)
        return df
    error_message = "错误：数据源调用成功，但未返回任何有效数据。"
    print(f"--- [数据接口] {error_message} ---")
    return error_message

def get_gold_forex_reserves(
    indicator_name: str,
    start_date: str,
    end_date: str
) -> Union[pd.DataFrame, str]:
    """
    获取央行黄金储备或国家外汇储备在【指定时间段内】的月度数据。
    
    :param indicator_name: 要查询的指标名称。必须是 "黄金储备" 或 "国家外汇储备" 之一。
    :param start_date: 开始日期, 格式 'YYYY-MM'.
    :param end_date: 结束日期, 格式 'YYYY-MM'.
    :return: 一个包含'统计时间'和指定指标列的DataFrame。
    """
    print(f"--- [数据获取] 正在获取 '{indicator_name}' 从 {start_date} 到 {end_date} 的数据... ---")
    VALID_INDICATORS = ["黄金储备", "国家外汇储备"]
    if indicator_name not in VALID_INDICATORS:
        error_message = f"错误: 无效的指标名称 '{indicator_name}'。有效选项: {VALID_INDICATORS}"
        print(f"错误: 无效的指标名称 '{indicator_name}'。有效选项: {VALID_INDICATORS}")
        return error_message
    df_or_error = _get_and_clean_gold_forex_data()
    if isinstance(df_or_error, str):
        print(f"--- [数据获取] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  # 直接将内部错误信息透传出去
    df = df_or_error
    if df.empty:
        error_message = "错误：数据获取成功，但未返回任何数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    try:
        mask = (df['统计时间'] >= start_date) & (df['统计时间'] <= end_date)
        filtered_df = df.loc[mask]
    except Exception as e:
        error_message = f"错误：在筛选日期 {start_date} 到 {end_date} 时发生错误: {e}。请检查日期格式是否为 YYYY-MM。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    if filtered_df.empty:
        error_message = f"错误：在 {start_date} 到 {end_date} 的时间范围内没有找到 '{indicator_name}' 的数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    try:
        return filtered_df[['统计时间', indicator_name]]
    except KeyError:
        error_message = f"错误：数据中未找到列 '{indicator_name}'。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message

def get_gold_forex_reserves_on_date(
    indicator_name: str,
    query_date: str
) -> Union[Dict[str, Any], str]:
    """
    获取央行黄金储备或国家外汇储备在【某一个指定月份】的（最近）有效数值。
    
    注意: 此函数将返回指定月份 'YYYY-MM' *或* 在此之前的 *最近一个* 有效数据点。
    
    :param indicator_name: 要查询的指标名称。必须是 "黄金储备" 或 "国家外汇储备" 之一。
    :param query_date: 要查询的月份, 格式 'YYYY-MM'.
    :return: 一个包含该月份利率数值的字典, 或一个错误信息字符串。
    """
    print(f"--- [数据获取] 正在获取 '{indicator_name}' 在 {query_date} (或之前) 的月度数据... ---")
    VALID_INDICATORS = ["黄金储备", "国家外汇储备"]
    if indicator_name not in VALID_INDICATORS:
        error_message = f"错误: 无效的指标名称 '{indicator_name}'。有效选项: {VALID_INDICATORS}"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    df_or_error = _get_and_clean_gold_forex_data()
    if isinstance(df_or_error, str):
        print(f"--- [数据获取] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    if df.empty:
        error_message = "错误：LPR数据获取成功，但未返回任何数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    if indicator_name not in df.columns:
        error_message = f"错误：数据源中未找到指标 '{indicator_name}'。可用指标: {df.columns.tolist()}"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    try:
        filtered_df = df[df['统计时间'] <= query_date]
        if filtered_df.empty:
            error_message = f"错误：在 {query_date} 或此日期之前没有找到任何数据。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        data_series = filtered_df.iloc[-1]
        value = data_series[indicator_name]
        actual_date = data_series['统计时间']
        if pd.isna(value):
            error_message = f"错误：在 {actual_date} 找到了数据行，但指标 '{indicator_name}' 的值为NaN。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        result_dict = {
            "indicator_name": indicator_name,
            "query_date": query_date,   
            "actual_date": actual_date,
            "value": value
        }
        return result_dict
    except Exception as e:
        error_message = f"错误: 筛选日期 {query_date} 时出错: {e}。请检查日期格式是否为 'YYYY-MM'。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message

def calculate_gold_forex_change(
    start_date: str,
    end_date: str
) -> Union[pd.DataFrame, str]: 
    """
    【分析工具】计算黄金储备和国家外汇储备在【指定时间段内】的累计涨跌幅。
    
    :param start_date: 开始日期, 格式 'YYYY-MM'.
    :param end_date: 结束日期, 格式 'YYYY-MM'.
    :return: 一个包含两个指标变化详情的DataFrame, 或一个错误信息字符串。
    """
    print(f"--- [分析任务] 正在计算黄金和外汇储备从 {start_date} 到 {end_date} 的变化... ---")
    df_or_error = _get_and_clean_gold_forex_data()
    if isinstance(df_or_error, str):
        print(f"--- [分析任务] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    try:
        mask = (df['统计时间'] >= start_date) & (df['统计时间'] <= end_date)
        filtered_df = df.loc[mask].sort_values('统计时间')
    except Exception as e:
        error_message = f"错误：在筛选日期 {start_date} 到 {end_date} 时发生错误: {e}。请检查日期格式是否为 YYYY-MM。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    if len(filtered_df) < 2:
        error_message = f"错误: 在时间范围 {start_date} 到 {end_date} 内数据点不足 (少于2个)，无法计算变化。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    try:
        start_data = filtered_df.iloc[0]
        end_data = filtered_df.iloc[-1]
        gold_start = start_data['黄金储备']
        gold_end = end_data['黄金储备']
        gold_change_pct = ((gold_end - gold_start) / gold_start) * 100 if gold_start != 0 else float('inf')
        forex_start = start_data['国家外汇储备']
        forex_end = end_data['国家外汇储备']
        forex_change_pct = ((forex_end - forex_start) / forex_start) * 100 if forex_start != 0 else float('inf')
    except KeyError as e:
        error_message = f"错误：数据中缺少必要的列: {e}。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    except Exception as e:
        error_message = f"错误：在计算变化时发生未知错误: {e}。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    result_data = {
        "指标名称": ["黄金储备 (万盎司)", "国家外汇储备 (亿美元)"],
        "开始时间": [start_data['统计时间'], start_data['统计时间']],
        "开始数值": [gold_start, forex_start],
        "结束时间": [end_data['统计时间'], end_data['统计时间']],
        "结束数值": [gold_end, forex_end],
        "变化百分比(%)": [round(gold_change_pct, 2), round(forex_change_pct, 2)]
    }
    return pd.DataFrame(result_data)

@lru_cache(maxsize=1)
def _get_and_clean_m2_data() -> Union[pd.DataFrame, str]:
    """
    [内部函数] 获取并预处理中国M2货币供应年率数据。
    [已修复] 仅使用 akshare 实时接口获取数据，移除所有本地 CSV 回退逻辑。
    """
    df = pd.DataFrame()
    
    try:
        print("--- [数据接口] 正在尝试主数据源 (akshare.macro_china_m2_yearly)... ---")
        
        # 1. 调用 API 获取数据
        df_ak = ak.macro_china_m2_yearly() 
        
        # 2. 标准化和精简
        df_ak.rename(columns={'日期': 'date', '今值': 'value'}, inplace=True)
        
        # 确保 date 列存在且可解析，以便后续操作
        if 'date' not in df_ak.columns or 'value' not in df_ak.columns:
             error_message = "错误：API 返回数据结构不匹配，缺少 '日期' 或 '今值' 列。"
             print(f"--- [数据接口] {error_message} ---")
             return error_message
             
        df_ak['date'] = pd.to_datetime(df_ak['date']).dt.strftime('%Y-%m')
        df = df_ak[['date', 'value']]
        
        print("--- [数据接口] 主数据源 (akshare接口) 获取成功。 ---")

    except Exception as e:
        error_message = f"错误：数据源调用失败 (akshare.macro_china_m2_yearly): {e}"
        print(f"--- [数据接口] {error_message} ---")
        return error_message
        
    # 3. 清洗和返回
    if not df.empty:
        df.dropna(subset=['value'], inplace=True)
        df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        df.sort_values('date', inplace=True, ignore_index=True)
        
        if df.empty:
            error_message = "错误：数据源获取成功，但清洗后(去重或去空值)未剩任何有效数据。"
            print(f"--- [数据接口] {error_message} ---")
            return error_message
            
        return df # 返回 DataFrame
        
    # 如果 df 为空但没有抛出异常
    error_message = "错误：数据源调用成功，但未返回任何有效数据。"
    print(f"--- [数据接口] {error_message} ---")
    return error_message

def get_m2_supply_rate(start_date: str, end_date: str) -> Union[pd.DataFrame, str]: 
    """
    获取中国M2货币供应年率在【指定时间段内】的月度数据。
    
    :param start_date: 开始日期, 格式 'YYYY-MM'.
    :param end_date: 结束日期, 格式 'YYYY-MM'.
    :return: 一个包含'date' (月份) 和 'value' (M2年率) 列的DataFrame, 或一个错误信息字符串。
    """
    print(f"--- [数据获取] 正在获取 M2供应年率 从 {start_date} 到 {end_date} 的数据... ---")
    df_or_error = _get_and_clean_m2_data()
    if isinstance(df_or_error, str):
        print(f"--- [数据获取] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    if df.empty:
        error_message = "错误：M2数据获取成功，但未返回任何数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    try:
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        filtered_df = df.loc[mask]
    except Exception as e:
        error_message = f"错误：在筛选日期 {start_date} 到 {end_date} 时发生错误: {e}。请检查日期格式是否为 YYYY-MM。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    if filtered_df.empty:
        error_message = f"错误：在 {start_date} 到 {end_date} 的时间范围内没有找到 M2 数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    return filtered_df

def get_m2_supply_rate_on_date(
    query_date: str
) -> Union[Dict[str, Any], str]: 
    """
    获取中国M2货币供应年率在【某一个指定月份】的（最近）有效数值。
    
    注意: 此函数将返回指定月份 'YYYY-MM' *或* 在此之前的 *最近一个* 有效数据点。
    
    :param query_date: 要查询的月份, 格式 'YYYY-MM'.
    :return: 一个包含该月份M2年率数值的字典, 或一个错误信息字符串。
    """
    print(f"--- [数据获取] 正在获取 M2供应年率 在 {query_date} (或之前) 的月度数据... ---")
    df_or_error = _get_and_clean_m2_data()
    if isinstance(df_or_error, str):
        print(f"--- [数据获取] 内部函数返回错误: {df_or_error} ---")
        return df_or_error 
    df = df_or_error
    if df.empty:
        error_message = "错误：M2数据获取成功，但未返回任何数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    try:
        filtered_df = df[df['date'] <= query_date]
        if filtered_df.empty:
            error_message = f"错误：在 {query_date} 或此日期之前没有找到任何M2数据。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        data_series = filtered_df.iloc[-1]
        value = data_series['value']
        actual_date = data_series['date'] 
        if pd.isna(value):
            error_message = f"错误：在 {actual_date} 找到了数据行，但 'value' (M2年率) 的值为NaN。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        result_dict = {
            "indicator_name": "M2货币供应年率",
            "query_date": query_date,   
            "actual_date": actual_date, 
            "value": value
        }
        return result_dict
    except Exception as e:
        error_message = f"错误: 筛选日期 {query_date} 时出错: {e}。请检查日期格式是否为 'YYYY-MM'。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message

def calculate_m2_supply_rate_change(
    start_date: str, 
    end_date: str
) -> Union[pd.DataFrame, str]: 
    """
    【分析工具】计算中国M2货币供应年率在【指定时间段内】的变化情况（绝对值变化）。
    
    :param start_date: 开始日期, 格式 'YYYY-MM'.
    :param end_date: 结束日期, 格式 'YYYY-MM'.
    :return: 一个包含M2年率变化详情的DataFrame, 或一个错误信息字符串。
    """
    print(f"--- [分析任务] 正在计算 M2供应年率 从 {start_date} 到 {end_date} 的变化... ---")
    df_or_error = _get_and_clean_m2_data()
    if isinstance(df_or_error, str):
        print(f"--- [分析任务] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    if df.empty:
        error_message = "错误：M2数据获取成功，但未返回任何数据。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    try:
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        filtered_df = df.loc[mask]
    except Exception as e:
        error_message = f"错误：在筛选日期 {start_date} 到 {end_date} 时发生错误: {e}。请检查日期格式是否为 YYYY-MM。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    if len(filtered_df) < 2:
        error_message = f"错误: 在时间范围 {start_date} 到 {end_date} 内数据点不足 (少于2个)，无法计算变化。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    try:
        start_data = filtered_df.iloc[0]
        end_data = filtered_df.iloc[-1]
        start_value = start_data['value']
        end_value = end_data['value']
        absolute_change = end_value - start_value
    except KeyError as e:
        error_message = f"错误：数据中缺少必要的 'value' 列: {e}。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    except Exception as e:
        error_message = f"错误：在计算变化时发生未知错误: {e}。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    result_data = {
        "指标名称": ["M2货币供应年率 (%)"],
        "开始时间": [start_data['date']],
        "开始数值": [start_value],
        "结束时间": [end_data['date']],
        "结束数值": [end_value],
        "绝对变化(百分点)": [round(absolute_change, 2)]
    }
    return pd.DataFrame(result_data)

@lru_cache(maxsize=1)
def _get_and_clean_currency_data() -> Union[pd.DataFrame, str]: 
    """
    [辅助函数] 获取并预处理人民币汇率中间价数据。
    使用缓存避免在单次运行中重复调用API。
    """
    print("--- [数据接口] 正在调用 'ak.currency_boc_safe'... ---")
    try:
        df = ak.currency_boc_safe()
        if df.empty:
            error_message = "错误：API (ak.currency_boc_safe) 调用成功，但未返回任何数据。"
            print(f"--- [数据接口] {error_message} ---")
            return error_message
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        return df
    except Exception as e:
        error_message = f"错误：调用API (ak.currency_boc_safe) 或处理数据时失败: {e}"
        print(f"--- [数据接口] {error_message} ---")
        return error_message 

def get_monthly_avg_currency_rate(
    currency_name: str,
    start_date: str,
    end_date: str
) -> Union[pd.DataFrame, str]: 
    """
    计算【某种外币】对人民币汇率中间价在【指定时间段内】的【月度算术平均值】。
    
    :param currency_name: 要查询的货币名称。必须是数据接口支持的币种之一，例如 "美元", "欧元", "日元" 等。
    :param start_date: 开始日期, 格式 'YYYY-MM-DD'.
    :param end_date: 结束日期, 格式 'YYYY-MM-DD'.
    :return: 一个包含'月份'和该货币月度平均汇率列的DataFrame, 或一个错误信息字符串。
    """
    print(f"--- [月度平均计算] 正在计算 '{currency_name}' 从 {start_date} 到 {end_date} 的月度平均汇率... ---")
    df_or_error = _get_and_clean_currency_data()
    if isinstance(df_or_error, str):
        print(f"--- [月度平均计算] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    if df.empty:
        error_message = "错误：汇率数据获取成功，但未返回任何数据。"
        print(f"--- [月度平均计算] {error_message} ---")
        return error_message
    if currency_name not in df.columns:
        valid_currencies = [col for col in df.columns if '代码' not in col] 
        error_message = f"错误: 无效的货币名称 '{currency_name}'。有效选项包括: {valid_currencies}"
        print(f"--- [月度平均计算] {error_message} ---")
        return error_message 
    try:
        filtered_df = df.loc[start_date:end_date]
    except Exception as e:
        error_message = f"错误: 筛选日期范围 {start_date} 到 {end_date} 时出错: {e}。请检查日期格式是否为 'YYYY-MM-DD'。"
        print(f"--- [月度平均计算] {error_message} ---")
        return error_message
    if filtered_df.empty:
        error_message = f"错误: 在时间范围 {start_date} 到 {end_date} 内没有找到 '{currency_name}' 的任何数据。"
        print(f"--- [月度平均计算] {error_message} ---")
        return error_message
    try:
        monthly_avg = filtered_df[currency_name].resample('M').mean()
        if monthly_avg.isnull().all():
            error_message = f"错误: 在 {start_date} 到 {end_date} 期间, '{currency_name}' 的数据全部无效 (NaN)，无法计算月度平均值。"
            print(f"--- [月度平均计算] {error_message} ---")
            return error_message
        monthly_avg.dropna(inplace=True)
    except Exception as e:
        error_message = f"错误: 在计算 '{currency_name}' 的月度平均值时发生错误: {e}"
        print(f"--- [月度平均计算] {error_message} ---")
        return error_message
    result_df = monthly_avg.reset_index()
    result_df.rename(columns={'日期': '月份', currency_name: f'{currency_name}_月度平均值'}, inplace=True)
    result_df['月份'] = result_df['月份'].dt.strftime('%Y-%m')
    return result_df

def get_currency_rate_on_date(
    currency_name: str,
    query_date: str
) -> Union[Dict[str, Any], str]: 
    """
    获取【某种外币】对人民币汇率中间价在【某一个指定日期】的（最近）有效数值。
    
    注意: 此函数将返回指定日期 'YYYY-MM-DD' *或* 在此之前的 *最近一个* 有效数据点。
    
    :param currency_name: 要查询的货币名称。必须是数据接口支持的币种之一，例如 "美元", "欧元", "日元" 等。
    :param query_date: 要查询的日期, 格式 'YYYY-MM-DD'.
    :return: 一个包含该日期汇率数值的字典, 或一个错误信息字符串。
    """
    print(f"--- [数据获取] 正在获取 '{currency_name}' 在 {query_date} (或之前) 的汇率数据... ---")
    df_or_error = _get_and_clean_currency_data()
    if isinstance(df_or_error, str):
        print(f"--- [数据获取] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    if df.empty:
        error_message = "错误：汇率数据获取成功，但未返回任何数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    if currency_name not in df.columns:
        valid_currencies = [col for col in df.columns if '代码' not in col]
        error_message = f"错误: 无效的货币名称 '{currency_name}'。有效选项包括: {valid_currencies}"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    try:
        data_series = df.asof(query_date)
        if data_series is None or pd.isna(data_series).all():
            error_message = f"错误：在 {query_date} 或此日期之前没有找到任何汇率数据。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        value = data_series[currency_name]
        actual_date = data_series.name.strftime('%Y-%m-%d') 
        if pd.isna(value):
            error_message = f"错误：在 {actual_date} 找到了数据行，但货币 '{currency_name}' 的值为NaN。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        result_dict = {
            "currency_name": currency_name,
            "query_date": query_date,   
            "actual_date": actual_date, 
            "value": value
        }
        return result_dict
    except Exception as e:
        error_message = f"错误: 筛选日期 {query_date} 时出错: {e}。请检查日期格式是否为 'YYYY-MM-DD'。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message

def get_stock_listing_date(
    name: Optional[str] = None,
    code: Optional[str] = None
) -> str: 
    """
    【信息检索工具】查询单只A股股票的上市时间。
    函数既可以接收股票代码(code)，也可以接收股票名称(name)作为输入。

    :param name: (可选) 股票的中文名称, 例如 "万科A"。
    :param code: (可选) 股票的代码, 例如 "000002"。name和code至少需要提供一个。
    :return: 股票的上市日期(格式 'YYYY-MM-DD'), 或者一个描述性的错误信息字符串。
    """
    print(f"--- [上市日期查询] 正在查询 '{name or code}' 的上市时间... ---")
    if not code and not name:
        error_message = "错误：必须提供股票代码 (code) 或股票名称 (name)。"
        print(f"--- [上市日期查询] {error_message} ---")
        return error_message # <-- 2. 修改点
    symbol = code
    if not symbol and name:
        print(f"--- [上市日期查询] 代码缺失, 正在通过名称 '{name}' 查找A股代码... ---")
        symbol = get_code_from_name(name=name, market='a') 
        if not symbol:
            error_message = f"错误：未能通过名称 '{name}' 找到对应的A股股票代码。"
            print(f"--- [上市日期查询] {error_message} ---")
            return error_message 
    symbol_cleaned = ''.join(filter(str.isdigit, str(symbol)))
    if not symbol_cleaned:
        error_message = f"错误：提供的代码 '{symbol}' 无效。"
        print(f"--- [上市日期查询] {error_message} ---")
        return error_message
    try:
        print(f"--- [上市日期查询] 正在调用API获取代码 '{symbol_cleaned}' 的详细信息... ---")
        info_df = ak.stock_individual_info_em(symbol=symbol_cleaned)
        if info_df.empty:
            error_message = f"错误：API未能返回代码 '{symbol_cleaned}' 的任何信息（可能是无效代码）。"
            print(f"--- [上市日期查询] {error_message} ---")
            return error_message
        info_dict = dict(zip(info_df['item'], info_df['value']))
        listing_date_str = info_dict.get('上市时间')
        if listing_date_str:
            listing_date = pd.to_datetime(str(listing_date_str), format='%Y%m%d').strftime('%Y-%m-%d')
            print(f"--- [上市日期查询] 成功找到 '{name or code}' 的上市时间: {listing_date} ---")
            return listing_date
        else:
            error_message = f"错误：在返回的数据中未能找到 '{name or code}' (代码: {symbol_cleaned}) 的上市时间字段。"
            print(f"--- [上市日期查询] {error_message} ---")
            return error_message 
    except Exception as e:
        error_message = f"错误：在为 '{name or code}' (代码: {symbol_cleaned}) 查询上市时间时发生API或处理错误: {e}"
        print(f"--- [上市日期查询] {error_message} ---")
        return error_message
    
@lru_cache(maxsize=1)
def _get_futures_symbol_map() -> Union[Dict[str, str], str]: # <-- 2. 更新了返回类型
    """
    [辅助函数] 获取新浪财经的主力连续合约品种列表，并创建一个名称到代码的映射。
    使用缓存避免重复调用API。
    """
    print("--- [数据接口] 正在调用 'ak.futures_display_main_sina' 获取期货品种列表... ---")
    try:
        df = ak.futures_display_main_sina()
        if df.empty:
            error_message = "错误：API (ak.futures_display_main_sina) 调用成功，但未返回任何期货品种数据。"
            print(f"--- [数据接口] {error_message} ---")
            return error_message
        name_to_symbol = {row['name'].replace('连续', ''): row['symbol'] for _, row in df.iterrows()}
        symbol_to_symbol = {row['symbol'].lower(): row['symbol'] for _, row in df.iterrows()}
        name_to_symbol.update(symbol_to_symbol)
        if not name_to_symbol:
            error_message = "错误：成功获取期货数据，但在创建映射时失败（结果为空）。"
            print(f"--- [数据接口] {error_message} ---")
            return error_message
        return name_to_symbol
    except Exception as e:
        error_message = f"错误：获取期货品种列表 (ak.futures_display_main_sina) 失败: {e}"
        print(f"--- [数据接口] {error_message} ---")
        return error_message 
    
def get_futures_history(
    identifier: str,
    start_date: str,
    end_date: str,
    columns_to_include: Optional[List[str]] = None
) -> Union[pd.DataFrame, str]:
    """
    获取【单个期货主力连续合约】在【指定时间段内】的日度历史行情数据。
    用户可以指定需要返回的数据列。
    
    :param identifier: 期货品种的代码或中文名称。例如 "IF0" 或 "沪深300指数期货"。
    :param start_date: 开始日期, 格式 'YYYY-MM-DD'.
    :param end_date: 结束日期, 格式 'YYYY-MM-DD'.
    :param columns_to_include: (可选) 一个包含所需数据列名的列表。
                            有效值: ['开盘价', '最高价', '最低价', '收盘价', '成交量', '持仓量', '动态结算价']。
                            默认为返回所有列。
    :return: 一个包含指定期货历史数据的 pandas.DataFrame 对象, 或一个错误信息字符串。
    """
    print(f"--- [期货查询] Manging '{identifier}' 从 {start_date} 到 {end_date} 的历史数据... ---")
    symbol_map_or_error = _get_futures_symbol_map()
    if isinstance(symbol_map_or_error, str):
        print(f"--- [期货查询] 内部函数返回错误: {symbol_map_or_error} ---")
        return symbol_map_or_error  
    symbol_map = symbol_map_or_error
    if not symbol_map: 
        error_message = "错误：无法获取期货品种列表（内部函数返回为空），无法继续查询。"
        print(f"--- [期货查询] {error_message} ---")
        return error_message
    normalized_identifier = identifier.lower().replace('连续', '')
    symbol = symbol_map.get(normalized_identifier)
    if not symbol:
        for name, sym in symbol_map.items():
            if isinstance(name, str) and normalized_identifier in name.lower():
                symbol = sym
                print(f"--- [期货查询] 模糊匹配到 '{name}' -> 代码 '{symbol}' ---")
                break
    if not symbol:
        error_message = f"错误: 未能识别期货品种 '{identifier}'。请检查代码或名称是否正确。有效选项（部分）: {list(symbol_map.keys())[:10]}..."
        print(f"--- [期货查询] {error_message} ---")
        return error_message
    DEFAULT_COLUMNS = ['开盘价', '最高价', '最低价', '收盘价', '成交量', '持仓量', '动态结算价']
    final_columns = DEFAULT_COLUMNS
    if columns_to_include:
        invalid_columns = [col for col in columns_to_include if col not in DEFAULT_COLUMNS]
        if invalid_columns:
            error_message = f"错误: 请求了无效的列名 {invalid_columns}。有效选项: {DEFAULT_COLUMNS}"
            print(f"--- [期货查询] {error_message} ---")
            return error_message
        final_columns = columns_to_include
    try:
        start_date_fmt = start_date.replace('-', '')
        end_date_fmt = end_date.replace('-', '')
        df = ak.futures_main_sina(symbol=symbol, start_date=start_date_fmt, end_date=end_date_fmt)
        if df.empty:
            error_message = f"信息: 在时间范围 {start_date} 到 {end_date} 内没有找到 '{identifier}' (代码: {symbol}) 的任何数据。"
            print(f"--- [期货查询] {error_message} ---")
            return error_message 
        missing_cols = ['日期'] + [col for col in final_columns if col not in df.columns]
        if len(missing_cols) > 1 or '日期' not in missing_cols:
            error_message = f"错误：API返回的数据中缺少请求的列: {missing_cols}。"
            print(f"--- [期货查询] {error_message} ---")
            return error_message
        return df[['日期'] + final_columns]
    except Exception as e:
        error_message = f"错误：在为 '{identifier}' (代码: {symbol}) 查询API时发生错误: {e}"
        print(f"--- [期货查询] {error_message} ---")
        return error_message 

@lru_cache(maxsize=1)
def _get_and_clean_ppi_data() -> Union[pd.DataFrame, str]: 
    """
    [内部函数] 获取并预处理中国PPI年率数据。
    使用缓存避免在单次运行中重复调用API。
    """
    print("--- [数据接口] 正在调用 'ak.macro_china_ppi_yearly'... ---")
    try:
        df = ak.macro_china_ppi_yearly()
        if df.empty:
            error_message = "错误：API (ak.macro_china_ppi_yearly) 调用成功，但未返回任何数据。"
            print(f"--- [数据接口] {error_message} ---")
            return error_message
        df.rename(columns={'日期': 'date', '今值': 'ppi_yoy'}, inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
        df = df[['date', 'ppi_yoy']].dropna()
        if df.empty:
            error_message = "错误：PPI数据获取成功，但清洗后(去空值)未剩任何有效数据。"
            print(f"--- [数据接口] {error_message} ---")
            return error_message
        df.sort_values('date', inplace=True)
        return df
    except Exception as e:
        error_message = f"错误：调用API (ak.macro_china_ppi_yearly) 或处理数据时失败: {e}"
        print(f"--- [数据接口] {error_message} ---")
        return error_message 
    
def get_ppi_yearly_by_quarters(
    quarters: List[str]
) -> Union[pd.DataFrame, str]: 
    """
    获取中国PPI年率在【一个或多个指定季度】内的【月度同比】数据。
    
    :param quarters: 一个包含一个或多个季度字符串的列表。格式必须是 'YYYYQX'，例如 ['2023Q1', '2023Q4']。
    :return: 一个包含'月份'和'ppi_yoy'(PPI月度同比)列的DataFrame, 或一个错误信息字符串。
    """
    print(f"--- [PPI查询] 正在获取 {quarters} 季度的月度PPI数据... ---")
    if not quarters:
        error_message = "错误: 'quarters' 列表不能为空。"
        print(f"--- [PPI查询] {error_message} ---")
        return error_message # <-- 3. 修改点
    try:
        start_month = "9999-99"
        end_month = "0000-00"
        for q in quarters:
            if not (isinstance(q, str) and len(q) == 6 and q[4] == 'Q' and q[5] in '1234'):
                raise ValueError(f"季度格式错误: '{q}' (必须是 'YYYYQX' 格式的字符串)")
            year = q[:4]
            quarter_num = int(q[5])
            q_start_month = f"{year}-{((quarter_num-1)*3)+1:02d}"
            q_end_month = f"{year}-{quarter_num*3:02d}"
            if q_start_month < start_month:
                start_month = q_start_month
            if q_end_month > end_month:
                end_month = q_end_month
    except (ValueError, TypeError, AttributeError) as e:
        error_message = f"错误: 解析季度列表时出错。请确保格式为 'YYYYQX'。错误详情: {e}"
        print(f"--- [PPI查询] {error_message} ---")
        return error_message 
    df_or_error = _get_and_clean_ppi_data()
    if isinstance(df_or_error, str):
        print(f"--- [PPI查询] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    if df.empty:
        error_message = "错误：PPI数据获取成功，但未返回任何数据。"
        print(f"--- [PPI查询] {error_message} ---")
        return error_message
    try:
        mask = (df['date'] >= start_month) & (df['date'] <= end_month)
        result_df = df.loc[mask].copy()
    except Exception as e:
        error_message = f"错误：在筛选日期 {start_month} 到 {end_month} 时发生错误: {e}。请检查日期格式是否为 YYYY-MM。"
        print(f"--- [PPI查询] {error_message} ---")
        return error_message
    if result_df.empty:
        error_message = f"错误：在 {start_month} 到 {end_month} 的总时间范围内没有找到任何PPI数据。"
        print(f"--- [PPI查询] {error_message} ---")
        return error_message
    def get_quarter(date_str):
        month = int(date_str.split('-')[1])
        return f"Q{ (month - 1) // 3 + 1 }"
    result_df['quarter'] = result_df['date'].apply(lambda x: f"{x[:4]}{get_quarter(x)}")
    final_df = result_df[result_df['quarter'].isin(quarters)]
    if final_df.empty:
        error_message = f"错误：在指定的季度 {quarters} 中未找到任何PPI数据（可能数据只存在于总范围内的其他月份）。"
        print(f"--- [PPI查询] {error_message} ---")
        return error_message
    return final_df[['date', 'ppi_yoy']].reset_index(drop=True)

def get_ppi_rate_on_date(
    query_date: str
) -> Union[Dict[str, Any], str]:
    """
    获取中国PPI年率在【某一个指定月份】的（最近）有效数值。
    
    注意: 此函数将返回指定月份 'YYYY-MM' *或* 在此之前的 *最近一个* 有效数据点。
    
    :param query_date: 要查询的月份, 格式 'YYYY-MM'.
    :return: 一个包含该月份PPI年率数值的字典, 或一个错误信息字符串。
    """
    print(f"--- [数据获取] 正在获取 PPI年率 在 {query_date} (或之前) 的月度数据... ---")
    df_or_error = _get_and_clean_ppi_data()
    if isinstance(df_or_error, str):
        print(f"--- [数据获取] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    if df.empty:
        error_message = "错误：PPI数据获取成功，但未返回任何数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    try:
        filtered_df = df[df['date'] <= query_date]
        if filtered_df.empty:
            error_message = f"错误：在 {query_date} 或此日期之前没有找到任何PPI数据。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        data_series = filtered_df.iloc[-1]
        value = data_series['ppi_yoy'] 
        actual_date = data_series['date']
        if pd.isna(value):
            error_message = f"错误：在 {actual_date} 找到了数据行，但 'ppi_yoy' 的值为NaN。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        result_dict = {
            "indicator_name": "PPI年率",
            "query_date": query_date, 
            "actual_date": actual_date, 
            "value": value
        }
        return result_dict
    except Exception as e:
        error_message = f"错误: 筛选日期 {query_date} 时出错: {e}。请检查日期格式是否为 'YYYY-MM'。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message

@lru_cache(maxsize=1)
def _get_and_clean_lpr_data() -> Union[pd.DataFrame, str]:
    """
    [辅助函数] 获取并预处理中国LPR品种数据。
    使用缓存避免在单次运行中重复调用API。
    """
    print("--- [数据接口] 正在调用 'ak.macro_china_lpr'... ---")
    try:
        df = ak.macro_china_lpr()
        if df.empty:
            error_message = "错误：API (ak.macro_china_lpr) 调用成功，但未返回任何LPR数据。"
            print(f"--- [数据接口] {error_message} ---")
            return error_message
        df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])
        df.set_index('TRADE_DATE', inplace=True)
        if df.index.empty:
            error_message = "错误：LPR数据获取成功，但处理后索引为空。"
            print(f"--- [数据接口] {error_message} ---")
            return error_message
        return df
    except Exception as e:
        error_message = f"错误：调用API (ak.macro_china_lpr) 或处理数据时失败: {e}"
        print(f"--- [数据接口] {error_message} ---")
        return error_message 

def get_lpr_rate(
    indicator_name: str,
    start_date: str,
    end_date: str
) -> Union[pd.DataFrame, str]: 
    """
    获取LPR（贷款市场报价利率）或其他相关贷款利率在【指定时间段内】的历史数据。
    
    :param indicator_name: 要查询的利率品种名称。必须是 "LPR1Y", "LPR5Y", "RATE_1", "RATE_2" 之一。
    :param start_date: 开始日期, 格式 'YYYY-MM-DD'.
    :param end_date: 结束日期, 格式 'YYYY-MM-DD'.
    :return: 一个包含'日期'和指定利率品种数值列的DataFrame, 或一个错误信息字符串。
    """
    print(f"--- [数据获取] 正在获取 '{indicator_name}' 从 {start_date} 到 {end_date} 的数据... ---")
    VALID_INDICATORS = ["LPR1Y", "LPR5Y", "RATE_1", "RATE_2"]
    if indicator_name not in VALID_INDICATORS:
        error_message = f"错误: 无效的指标名称 '{indicator_name}'。有效选项: {VALID_INDICATORS}"
        print(f"--- [数据获取] {error_message} ---")
        return error_message 
    df_or_error = _get_and_clean_lpr_data()
    if isinstance(df_or_error, str):
        print(f"--- [数据获取] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    if df.empty:
        error_message = "错误：LPR数据获取成功，但未返回任何数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    try:
        if indicator_name not in df.columns:
            error_message = f"错误：LPR数据源中未找到指标 '{indicator_name}'。可用指标: {df.columns.tolist()}"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        filtered_df = df.loc[start_date:end_date, [indicator_name]].dropna().reset_index()
        filtered_df.rename(columns={'TRADE_DATE': '日期'}, inplace=True)
    except Exception as e:
        error_message = f"错误: 筛选时出错（日期 {start_date} 到 {end_date} 或指标 '{indicator_name}'）。请检查日期格式是否为 'YYYY-MM-DD'。错误详情: {e}"
        print(f"--- [数据获取] {error_message} ---")
        return error_message 
    if filtered_df.empty:
        error_message = f"错误：在 {start_date} 到 {end_date} 的时间范围内没有找到 '{indicator_name}' 的数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    return filtered_df

def get_lpr_rate_on_date(
    indicator_name: str,
    query_date: str
) -> Union[Dict[str, Any], str]: 
    """
    获取LPR（贷款市场报价利率）在【某一个指定日期】的（最近）有效数值。
    
    注意: 此函数将返回指定日期 *或* 在此之前的 *最近一个* 有效数据点。
    
    :param indicator_name: 要查询的利率品种名称。必须是 "LPR1Y", "LPR5Y", "RATE_1", "RATE_2" 之一。
    :param query_date: 要查询的日期, 格式 'YYYY-MM-DD'.
    :return: 一个包含该日期利率数值的字典, 或一个错误信息字符串。
    """
    print(f"--- [数据获取] 正在获取 '{indicator_name}' 在 {query_date} (或之前) 的数据... ---")
    VALID_INDICATORS = ["LPR1Y", "LPR5Y", "RATE_1", "RATE_2"]
    if indicator_name not in VALID_INDICATORS:
        error_message = f"错误: 无效的指标名称 '{indicator_name}'。有效选项: {VALID_INDICATORS}"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    df_or_error = _get_and_clean_lpr_data()
    if isinstance(df_or_error, str):
        print(f"--- [数据获取] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    if df.empty:
        error_message = "错误：LPR数据获取成功，但未返回任何数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    if indicator_name not in df.columns:
        error_message = f"错误：LPR数据源中未找到指标 '{indicator_name}'。可用指标: {df.columns.tolist()}"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    try:
        data_series = df.asof(query_date)
        if data_series is None or pd.isna(data_series).all():
            error_message = f"错误：在 {query_date} 或此日期之前没有找到任何LPR数据。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        value = data_series[indicator_name]
        actual_date = data_series.name.strftime('%Y-%m-%d') 
        if pd.isna(value):
            error_message = f"错误：在 {actual_date} 找到了数据行，但指标 '{indicator_name}' 的值为NaN。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        result_dict = {
            "indicator_name": indicator_name,
            "query_date": query_date,
            "actual_date": actual_date,
            "value": value
        }
        return result_dict
    except Exception as e:
        error_message = f"错误: 筛选日期 {query_date} 时出错: {e}。请检查日期格式是否为 'YYYY-MM-DD'。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message

def calculate_lpr_rate_change(
    indicator_name: str,
    start_date: str,
    end_date: str
) -> Union[pd.DataFrame, str]: 
    """
    计算某个LPR利率品种在【指定时间段内】的变化值（单位：基点）。
    
    :param indicator_name: 要分析的利率品种名称。必须是 "LPR1Y", "LPR5Y", "RATE_1", "RATE_2" 之一。
    :param start_date: 开始日期, 格式 'YYYY-MM-DD'.
    :param end_date: 结束日期, 格式 'YYYY-MM-DD'.
    :return: 一个包含利率变化详情的DataFrame, 或一个错误信息字符串。
    """
    print(f"--- [分析任务] 正在计算 '{indicator_name}' 从 {start_date} 到 {end_date} 的变化... ---")
    df_or_error = get_lpr_rate(indicator_name, start_date, end_date)
    if isinstance(df_or_error, str):
        print(f"--- [分析任务] 内部数据获取失败: {df_or_error} ---")
        return df_or_error  
    filtered_df = df_or_error
    if len(filtered_df) < 2:
        error_message = f"错误: 在时间范围 {start_date} 到 {end_date} 内数据点不足 (少于2个)，无法计算变化。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message 
    try:
        start_data = filtered_df.iloc[0]
        end_data = filtered_df.iloc[-1]
        start_value = start_data[indicator_name]
        end_value = end_data[indicator_name]
        basis_point_change = (end_value - start_value) * 100
    except KeyError as e:
        error_message = f"错误：在分析数据时缺少必要的列: {e}。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    except Exception as e:
        error_message = f"错误：在计算变化时发生未知错误: {e}。"
        print(f"--- [分析任务] {error_message} ---")
        return error_message
    result_data = {
        "指标名称": [indicator_name],
        "开始时间": [start_data['日期'].strftime('%Y-%m-%d')],
        "开始数值(%)": [start_value],
        "结束时间": [end_data['日期'].strftime('%Y-%m-%d')],
        "结束数值(%)": [end_value],
        "变化(基点)": [int(basis_point_change)]
    }
    return pd.DataFrame(result_data)

def calculate_bond_yield_change_on_date(
    target_date: str,
    curve_name: str = "中债国债收益率曲线",
    terms: Optional[List[str]] = None
) -> Union[pd.DataFrame, str]: 
    """
    【分析工具】计算并统计在【某一个指定日期】或【其后的首个交易日】，特定债券收益率曲线上多个期限的收益率及其相比上一交易日的变化幅度（单位：基点BP）。
    [最终修正版] 如果用户未指定期限(terms)，函数将自动分析所有可用的期限。
    
    :param target_date: 要查询的目标日期, 格式 'YYYY-MM-DD'.
    :param curve_name: (可选) 要查询的曲线名称。默认为 "中债国债收益率曲线"。
    :param terms: (可选) 一个包含所需期限的列表。如果未提供，则自动分析所有可用期限。
    :return: 一个包含各期限收益率及变化幅度的DataFrame, 或一个错误信息字符串。
    """
    print(f"--- [收益率变化分析] 正在分析 '{curve_name}' 在 {target_date} 附近的变化... ---")
    try:
        base_dt = pd.to_datetime(target_date)
        start_dt, end_dt = base_dt - timedelta(days=10), base_dt + timedelta(days=7)
        start_date_fmt, end_date_fmt = start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d')
        print(f"--- [数据接口] 正在获取 {start_date_fmt} 到 {end_date_fmt} 的债券收益率数据... ---")
        df = ak.bond_china_yield(start_date=start_date_fmt, end_date=end_date_fmt)
        if df.empty:
            error_message = f"错误：API (ak.bond_china_yield) 在 {start_date_fmt} 到 {end_date_fmt} 范围内未返回任何数据。"
            print(f"--- [数据接口] {error_message} ---")
            return error_message
        df_filtered = df[df['曲线名称'] == curve_name].copy()
        if df_filtered.empty:
            error_message = f"错误: 未能找到曲线名称为 '{curve_name}' 的数据。"
            print(f"--- [收益率变化分析] {error_message} ---")
            return error_message 
        df_filtered['日期'] = pd.to_datetime(df_filtered['日期'])
        df_filtered.sort_values('日期', inplace=True)
        today_or_next_trade_day = df_filtered[df_filtered['日期'] >= base_dt]
        if today_or_next_trade_day.empty:
            error_message = f"错误: 未能找到 {target_date} 或其后的任何交易日数据。"
            print(f"--- [收益率变化分析] {error_message} ---")
            return error_message 
        actual_today_series = today_or_next_trade_day.iloc[0]
        actual_today_date = actual_today_series['日期']
        prev_trade_days = df_filtered[df_filtered['日期'] < actual_today_date]
        if prev_trade_days.empty:
            error_message = f"错误: 未能找到 {actual_today_date.strftime('%Y-%m-%d')} 之前的交易日数据，无法计算变化。"
            print(f"--- [收益率变化分析] {error_message} ---")
            return error_message 
        prev_series = prev_trade_days.iloc[-1]
        print(f"--- [智能日期定位] 使用实际交易日: {actual_today_date.strftime('%Y-%m-%d')} 与其上一个交易日: {prev_series['日期'].strftime('%Y-%m-%d')} 进行比较。 ---")
        all_possible_terms = ['3月', '6月', '1年', '3年', '5年', '7年', '10年', '30年']
        terms_to_analyze = []
        if terms: 
            terms_to_analyze = terms
        else: 
            print("--- [期限分析] 用户未指定期限，将自动分析所有可用期限。 ---")
            terms_to_analyze = [term for term in all_possible_terms if term in df_filtered.columns]
        if not terms_to_analyze:
            error_message = "错误：未能确定任何要分析的期限。用户未指定，且自动检测也未发现任何标准期限。"
            print(f"--- [收益率变化分析] {error_message} ---")
            return error_message
        results = []
        for term in terms_to_analyze:
            if term not in df_filtered.columns:
                print(f"警告: 请求的期限 '{term}' 不存在于数据中，已跳过。")
                continue
            today_yield = actual_today_series[term]
            prev_yield = prev_series[term]
            change_in_bp = (today_yield - prev_yield) * 100 if pd.notna(today_yield) and pd.notna(prev_yield) else None
            results.append({
                "期限": term,
                f"{actual_today_date.strftime('%Y-%m-%d')}收益率(%)": today_yield,
                f"{prev_series['日期'].strftime('%Y-%m-%d')}收益率(%)": prev_yield,
                "变化幅度(基点)": change_in_bp
            })
        if not results:
            error_message = f"错误：分析完成，但未生成任何有效结果。请检查请求的期限 {terms} 是否有效。"
            print(f"--- [收益率变化分析] {error_message} ---")
            return error_message
        return pd.DataFrame(results)
    except Exception as e:
        error_message = f"错误：[收益率变化分析] 执行过程中发生未知错误: {e}"
        print(error_message)
        return error_message 

def get_rmb_central_parity_rate_change(
    start_date: str, 
    end_date: str, 
    currencies: List[str]
) -> Union[pd.DataFrame, str]:
    """
    获取并计算指定日期范围内、特定币种的人民币汇率中间价每日涨跌幅。

    Args:
        start_date (str): 查询的开始日期，格式应为 'YYYY-MM-DD'。
        end_date (str): 查询的结束日期，格式应为 'YYYY-MM-DD'。
        currencies (List[str]): 一个包含所需货币标准名称的列表。
                                例如: ['美元', '欧元', '泰铢']。

    Returns:
        Union[pd.DataFrame, str]: 
            - 成功: 一个包含指定货币在指定日期范围内每日涨跌幅的DataFrame。
            - 失败: 一个包含错误信息的字符串。                      
    注意:
        - 正值表示当日汇率数值升高，负值表示降低。
        - 其经济学含义需根据标价法解释。
    """
    if not isinstance(currencies, list) or not currencies:
        return "错误：'currencies' 参数必须是一个非空的列表，例如 ['美元', '欧元']。"
    try:
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
    except ValueError:
        return "错误：日期格式不正确，请使用 'YYYY-MM-DD' 格式。"
    try:
        currency_df = ak.currency_boc_safe()
        if currency_df.empty:
            error_message = "错误：API (ak.currency_boc_safe) 调用成功，但未返回任何数据。"
            print(f"--- [汇率涨跌幅] {error_message} ---")
            return error_message
        currency_df['日期'] = pd.to_datetime(currency_df['日期'])
        currency_df = currency_df.set_index('日期').sort_index()
        valid_currencies = [c for c in currencies if c in currency_df.columns]
        missing_currencies = [c for c in currencies if c not in currency_df.columns]
        if missing_currencies:
            print(f"警告：以下货币名称无法找到，将被忽略: {', '.join(missing_currencies)}")
        if not valid_currencies:
            available_currencies = [c for c in currency_df.columns if '代码' not in c]
            error_message = f"错误：您输入的所有货币名称 {currencies} 都不在可用列表中。可用选项（部分）: {available_currencies[:10]}..."
            print(f"--- [汇率涨跌幅] {error_message} ---")
            return error_message
        filtered_df = currency_df[valid_currencies].copy()
        for col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
        daily_change_df = filtered_df.pct_change() * 100
        final_df = daily_change_df.loc[start_date:end_date].copy()
        final_df.columns = [f"{col}_涨跌幅" for col in final_df.columns]
        final_df.dropna(how='all', inplace=True)
        if final_df.empty:
            error_message = f"错误：在 {start_date} 到 {end_date} 的时间范围内没有找到您所选货币 {valid_currencies} 的任何涨跌幅数据。"
            print(f"--- [汇率涨跌幅] {error_message} ---")
            return error_message
        return final_df
    except Exception as e:
        error_message = f"错误：[汇率涨跌幅] 处理过程中发生未知错误: {e}"
        print(error_message)
        return error_message

def get_currency_rate_on_date(
    currency_name: str,
    query_date: str
) -> Union[Dict[str, Any], str]:
    """
    获取【某种外币】对人民币汇率中间价在【某一个指定日期】的（最近）有效数值。
    
    注意: 此函数将返回指定日期 'YYYY-MM-DD' *或* 在此之前的 *最近一个* 有效数据点。
    
    :param currency_name: 要查询的货币名称。必须是数据接口支持的币种之一，例如 "美元", "欧元", "日元" 等。
    :param query_date: 要查询的日期, 格式 'YYYY-MM-DD'.
    :return: 一个包含该日期汇率数值的字典, 或一个错误信息字符串。
    """
    print(f"--- [数据获取] 正在获取 '{currency_name}' 在 {query_date} (或之前) 的汇率数据... ---")
    df_or_error = _get_and_clean_currency_data()
    if isinstance(df_or_error, str):
        print(f"--- [数据获取] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    if df.empty:
        error_message = "错误：汇率数据获取成功，但未返回任何数据。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    if currency_name not in df.columns:
        valid_currencies = [col for col in df.columns if '代码' not in col] 
        error_message = f"错误: 无效的货币名称 '{currency_name}'。有效选项包括: {valid_currencies}"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    try:
        data_series = df.asof(query_date)
        if data_series is None or pd.isna(data_series).all():
            error_message = f"错误：在 {query_date} 或此日期之前没有找到任何汇率数据。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        value = data_series[currency_name]
        actual_date = data_series.name.strftime('%Y-%m-%d') 
        if pd.isna(value):
            error_message = f"错误：在 {actual_date} 找到了数据行，但货币 '{currency_name}' 的值为NaN。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        result_dict = {
            "currency_name": currency_name,
            "query_date": query_date,   
            "actual_date": actual_date, 
            "value": value
        }
        return result_dict
    except Exception as e:
        error_message = f"错误: 筛选日期 {query_date} 时出错: {e}。请检查日期格式是否为 'YYYY-MM-DD'。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message

_forex_hist_cache: Dict[str, pd.DataFrame] = {}

def _get_and_clean_forex_data(symbol: str) -> Union[pd.DataFrame, str]:
    """
    [辅助函数] 获取并缓存单个外汇品种的 *全部* 历史数据。
    """
    global _forex_hist_cache
    if symbol in _forex_hist_cache:
        print(f"--- [外汇缓存] 命中: 正在从缓存中读取 '{symbol}'... ---")
        return _forex_hist_cache[symbol]
    print(f"--- [外汇API] 缓存未命中: 正在调用 ak.forex_hist_em(symbol='{symbol}')... ---")
    try:
        df = ak.forex_hist_em(symbol=symbol)
        if df.empty:
            error_message = f"错误: API (ak.forex_hist_em) 为 '{symbol}' 返回了空数据。"
            print(f"--- [外汇API] {error_message} ---")
            return error_message
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        _forex_hist_cache[symbol] = df
        return df
    except Exception as e:
        error_message = f"错误: 调用API (ak.forex_hist_em) 或处理 '{symbol}' 数据时失败: {e}"
        print(f"--- [外汇API] {error_message} ---")
        return error_message

def get_forex_price_on_date(
    symbol: str,
    query_date: str
) -> Union[Dict[str, Any], str]: 
    """
    获取【单个外汇品种】在【某一个指定日期】的（最近）有效行情数据。
    [原子化工具]
    
    注意: 此函数将返回指定日期 'YYYY-MM-DD' *或* 在此之前的 *最近一个* 有效数据点。
    
    :param symbol: 要查询的品种代码。例如: 'USDCNH', 'EURCNH'。
    :param query_date: 要查询的日期, 格式 'YYYY-MM-DD'.
    :return: 一个包含该日期行情数值的字典, 或一个错误信息字符串。
    """
    print(f"--- [数据获取] 正在获取 '{symbol}' 在 {query_date} (或之前) 的数据... ---")
    df_or_error = _get_and_clean_forex_data(symbol)
    if isinstance(df_or_error, str):
        print(f"--- [数据获取] 内部函数返回错误: {df_or_error} ---")
        return df_or_error  
    df = df_or_error
    try:
        data_series = df.asof(query_date)
        if data_series is None or pd.isna(data_series).all():
            error_message = f"错误：在 {query_date} 或此日期之前没有找到 '{symbol}' 的任何数据。"
            print(f"--- [数据获取] {error_message} ---")
            return error_message
        result_dict = data_series.to_dict()
        result_dict['symbol'] = symbol 
        result_dict['query_date'] = query_date
        result_dict['actual_date'] = data_series.name.strftime('%Y-%m-%d') 
        if '日期' in result_dict:
            del result_dict['日期']
        return result_dict
    except Exception as e:
        error_message = f"错误: 筛选日期 {query_date} 时出错: {e}。请检查日期格式是否为 'YYYY-MM-DD'。"
        print(f"--- [数据获取] {error_message} ---")
        return error_message
    
def get_forex_history_em(
    symbols: List[str],
    start_date: str,
    end_date: str,
    columns_to_include: Optional[List[str]] = None
) -> Union[pd.DataFrame, str]:
    """
    从东方财富网获取一个或多个外汇品种在指定时间段内的历史行情数据。
    """
    if not isinstance(symbols, list) or not symbols:
        error_message = "错误：'symbols' 参数必须是一个非空的列表，例如 ['USDCNH']。"
        print(f"--- [外汇历史] {error_message} ---")
        return error_message
    VALID_COLUMNS = {'最新价', '涨跌额', '涨跌幅', '今开', '最高', '最低', '昨收'}
    if columns_to_include:
        invalid_columns = [col for col in columns_to_include if col not in VALID_COLUMNS]
        if invalid_columns:
            error_message = f"错误：请求了无效的列名 {invalid_columns}。有效列名包括: {list(VALID_COLUMNS)}"
            print(f"--- [外汇历史] {error_message} ---")
            return error_message
    else:
        columns_to_include = list(VALID_COLUMNS)
    try:
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
    except ValueError:
        error_message = "错误：日期格式不正确，请使用 'YYYY-MM-DD' 格式。"
        print(f"--- [外汇历史] {error_message} ---")
        return error_message
    all_dfs = []
    errors = []
    print(f"--- [外汇历史] 开始获取 {symbols} 从 {start_date} 到 {end_date} 的数据...")
    for symbol in symbols:
        df_or_error = _get_and_clean_forex_data(symbol)
        if isinstance(df_or_error, pd.DataFrame):
            all_dfs.append(df_or_error)
        else:
            errors.append(f"'{symbol}': 获取失败，错误: {df_or_error}")
    if not all_dfs:
        error_message = f"错误：未能成功获取任何一个品种的数据。详情: {'; '.join(errors)}"
        print(f"--- [外汇历史] {error_message} ---")
        return error_message
    if errors:
        print(f"--- [外汇历史] 警告：部分品种获取失败: {'; '.join(errors)} ---")
    try:
        combined_df = pd.concat(all_dfs, ignore_index=False) 
        mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
        filtered_df = combined_df.loc[mask].copy()
        if filtered_df.empty:
            error_message = f"错误：在指定日期范围 {start_date} 到 {end_date} 内没有找到任何数据。"
            print(f"--- [外汇历史] {error_message} ---")
            return error_message
        missing_cols = [col for col in columns_to_include if col not in filtered_df.columns]
        if '代码' not in filtered_df.columns:
            missing_cols.append('代码')
        if missing_cols:
            error_message = f"错误：API返回的数据中缺少必要的列: {missing_cols}。"
            print(f"--- [外汇历史] {error_message} ---")
            return error_message
        pivot_df = filtered_df.pivot_table(
            index=filtered_df.index,
            columns='代码',
            values=columns_to_include
        )
        return pivot_df
    except Exception as e:
        error_message = f"错误：在处理和重塑数据时发生错误: {e}"
        print(f"--- [外汇历史] {error_message} ---")
        return error_message
    
_INDEX_DATA_CACHE: Optional[pd.DataFrame] = None
PICKLE_INDEX_KEY = 'index_stock_info'
@lru_cache(maxsize=1)
def _load_index_data() -> Union[pd.DataFrame, str]:
    global _INDEX_DATA_CACHE
    if _INDEX_DATA_CACHE is not None:
        if isinstance(_INDEX_DATA_CACHE, pd.DataFrame):
            return _INDEX_DATA_CACHE
        print("--- [缓存警告] 缓存中为错误信息, 尝试重新加载... ---")
    print(f"--- [数据加载] 正在从 Pickle 文件 '{LOCAL_PICKLE_FILE}' 加载 '{PICKLE_INDEX_KEY}'... ---")
    if not os.path.exists(LOCAL_PICKLE_FILE):
        error_message = f"错误: Pickle 文件未找到! 路径: {LOCAL_PICKLE_FILE}"
        print(f"--- [数据加载] {error_message} ---")
        _INDEX_DATA_CACHE = error_message
        return error_message 
    try:
        with open(LOCAL_PICKLE_FILE, 'rb') as f:
            data_archive = pickle.load(f)
        if PICKLE_INDEX_KEY not in data_archive:
            error_message = f"错误: Pickle 归档中缺少键名 '{PICKLE_INDEX_KEY}' 的数据集。"
            print(f"--- [数据加载] {error_message} ---")
            _INDEX_DATA_CACHE = error_message
            return error_message
        df = data_archive[PICKLE_INDEX_KEY]
        if df.empty:
            error_message = f"错误：数据集 '{PICKLE_INDEX_KEY}' 加载成功，但 DataFrame 为空。"
            print(f"--- [数据加载] {error_message} ---")
            _INDEX_DATA_CACHE = error_message
            return error_message
        if 'index_code' in df.columns:
            df['index_code'] = df['index_code'].astype(str)
        df = df.set_index('display_name')
        _INDEX_DATA_CACHE = df
        print(f"成功加载并缓存索引数据: {PICKLE_INDEX_KEY}")
        return _INDEX_DATA_CACHE
    except KeyError as ke:
        error_message = f"错误: 文件中缺少必需的列: {ke}。"
        print(f"--- [数据加载] {error_message} ---")
        _INDEX_DATA_CACHE = error_message 
        return error_message
    except Exception as e:
        error_message = f"加载索引数据时发生未知错误: {e}"
        print(f"--- [数据加载] {error_message} ---")
        _INDEX_DATA_CACHE = error_message 
        return error_message

InfoType = Literal["code", "publish_date"]

def get_index_stock_info(index_name: str, info_type: InfoType) -> str: 
    """
    根据指数的中文名称, 从本地CSV文件获取其代码或发布日期。
    
    Returns:
        str: 成功时返回找到的信息 (代码或日期), 失败时返回一个错误信息字符串。
    """
    df_or_error = _load_index_data()
    if isinstance(df_or_error, str):
        print(f"--- [索引查询] 内部数据加载失败: {df_or_error} ---")
        return df_or_error  
    df_cache = df_or_error
    target_column = ""
    if info_type == "code":
        target_column = "index_code"
    elif info_type == "publish_date":
        target_column = "publish_date"
    else:
        error_message = f"错误: 无效的 info_type '{info_type}'. 必须是 'code' 或 'publish_date'。"
        print(f"--- [索引查询] {error_message} ---")
        return error_message 
    try:
        result = df_cache.loc[index_name, target_column]
        return str(result) 
    except KeyError:
        error_message = f"错误: 未能在本地文件中找到名称为 '{index_name}' 的指数。"
        print(f"--- [索引查询] {error_message} ---")
        return error_message
    except Exception as e:
        error_message = f"查找 '{index_name}' 数据时发生未知错误: {e}"
        print(f"--- [索引查询] {error_message} ---")
        return error_message 
    
def get_index_components(
    symbol: str, 
    include_weights: bool = False
) -> Union[List[Dict[str, Any]], str]: 
    """
    根据中证指数代码, 获取其所有成分股列表, 并可选择性地包含其权重。

    Args:
        symbol (str): 指数代码, 例如 "000300" (沪深300) 或 "000688.SH" (科创50)。
                    函数会自动处理 .SH 或 .SZ 等后缀。
        include_weights (bool, optional): 是否获取权重。
                                        True: 返回包含权重的列表 (调用 index_stock_cons_weight_csindex)。
                                        False: 仅返回成分股列表 (调用 index_stock_cons_csindex)。
                                        默Renault 为 False。

    Returns:
        Union[List[Dict[str, Any]], str]: 
            - 成功: 返回一个列表, 每个元素是一个字典。
            - if include_weights=False: [{"stock_code": "...", "stock_name": "..."}]
            - if include_weights=True: [{"stock_code": "...", "stock_name": "...", "weight": 0.52}]
            - 失败: (例如代码无效或网络问题) 返回一个错误信息字符串。
    """
    if not symbol:
        error_message = "错误：'symbol' 参数不能为空。"
        print(f"--- [成分股查询] {error_message} ---")
        return error_message
    try:
        clean_symbol = str(symbol).split('.')[0]
        if not clean_symbol:
            error_message = f"错误：提供的 'symbol' ('{symbol}') 无效。"
            print(f"--- [成分股查询] {error_message} ---")
            return error_message
        result_df = pd.DataFrame()
        output_columns = []
        if include_weights:
            print(f"Tool call: calling ak.index_stock_cons_weight_csindex(symbol='{clean_symbol}')")
            df = ak.index_stock_cons_weight_csindex(symbol=clean_symbol)
            if df.empty or "成分券代码" not in df.columns or "权重" not in df.columns:
                error_message = f"错误：权重接口 (ak.index_stock_cons_weight_csindex) 为 '{clean_symbol}' 返回了空数据或无效数据（缺少'成分券代码'或'权重'列）。"
                print(f"--- [成分股查询] {error_message} ---")
                return error_message 
            df_renamed = df.rename(columns={
                "成分券代码": "stock_code",
                "成分券名称": "stock_name",
                "权重": "weight"
            })
            output_columns = ["stock_code", "stock_name", "weight"]
            result_df = df_renamed[output_columns]
        else:
            print(f"Tool call: calling ak.index_stock_cons_csindex(symbol='{clean_symbol}')")
            df = ak.index_stock_cons_csindex(symbol=clean_symbol)
            if df.empty or "成分券代码" not in df.columns:
                error_message = f"错误：成分股接口 (ak.index_stock_cons_csindex) 为 '{clean_symbol}' 返回了空数据或无效数据（缺少'成分券代码'列）。"
                print(f"--- [成分股查询] {error_message} ---")
                return error_message 
            df_renamed = df.rename(columns={
                "成分券代码": "stock_code",
                "成分券名称": "stock_name"
            })
            output_columns = ["stock_code", "stock_name"]
            result_df = df_renamed[output_columns]
        if result_df.empty:
            error_message = f"错误：数据在重命名的过程中丢失，'{clean_symbol}' 未返回有效成分股。"
            print(f"--- [成分股查询] {error_message} ---")
            return error_message
        return result_df.to_dict('records') 
    except Exception as e:
        error_message = f"错误: 在为 '{symbol}' (clean: '{clean_symbol}') 调用 akshare 接口时发生异常: {e}"
        print(f"--- [成分股查询] {error_message} ---")
        return error_message 
    
PICKLE_KEY_A = 'a_name_code'
PICKLE_KEY_HK = 'hk_name_code'
PICKLE_KEY_US = 'us_name_code_market'

@lru_cache(maxsize=1)
def _load_pickle_archive() -> Union[Dict[str, pd.DataFrame], str]:
    if not os.path.exists(LOCAL_PICKLE_FILE):
        return f"错误: Pickle 文件未找到! 路径: {LOCAL_PICKLE_FILE}"
    try:
        with open(LOCAL_PICKLE_FILE, 'rb') as f:
            data_archive = pickle.load(f)
        if not isinstance(data_archive, dict):
            return "错误: Pickle 文件加载成功，但内容不是预期的字典类型。"
        print(f"--- [Cache] 成功加载并缓存 Pickle 归档 ({len(data_archive)} 个数据集)。 ---")
        return data_archive
    except Exception as e:
        return f"错误: 加载 Pickle 文件时出错: {e}"
    
@lru_cache(maxsize=1)
def _load_a_stock_data() -> Union[pd.DataFrame, str]:
    archive = _load_pickle_archive()
    if isinstance(archive, str): return archive
    if PICKLE_KEY_A not in archive:
        return f"错误: Pickle 归档中缺少键名 '{PICKLE_KEY_A}' 的数据集。"
    df = archive[PICKLE_KEY_A]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return f"错误: Pickle 归档中 '{PICKLE_KEY_A}' 数据集为空或类型错误。"
    if not {'代码', '名称'}.issubset(df.columns):
        return f"错误: A股数据集 ('{PICKLE_KEY_A}') 缺少 '代码' 或 '名称' 列。"
    print(f"--- [Cache] A股数据加载成功。")
    return df

@lru_cache(maxsize=1)
def _load_hk_stock_data() -> Union[pd.DataFrame, str]:
    archive = _load_pickle_archive()
    if isinstance(archive, str): return archive 
    if PICKLE_KEY_HK not in archive:
        return f"错误: Pickle 归档中缺少键名 '{PICKLE_KEY_HK}' 的数据集。"
    df = archive[PICKLE_KEY_HK]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return f"错误: Pickle 归档中 '{PICKLE_KEY_HK}' 数据集为空或类型错误。"
    if not {'代码', '中文名称'}.issubset(df.columns):
        return f"错误: 港股数据集 ('{PICKLE_KEY_HK}') 缺少 '代码' 或 '中文名称' 列。"
    print(f"--- [Cache] 港股数据加载成功。")
    return df

@lru_cache(maxsize=1)
def _load_us_stock_data() -> Union[pd.DataFrame, str]:
    archive = _load_pickle_archive()
    if isinstance(archive, str): return archive 
    if PICKLE_KEY_US not in archive:
        return f"错误: Pickle 归档中缺少键名 '{PICKLE_KEY_US}' 的数据集。"
    df = archive[PICKLE_KEY_US]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return f"错误: Pickle 归档中 '{PICKLE_KEY_US}' 数据集为空或类型错误。"
    if not {'name', 'symbol', 'market'}.issubset(df.columns):
        return f"错误: 美股数据集 ('{PICKLE_KEY_US}') 缺少 'name', 'symbol', 或 'market' 列。"
    print(f"--- [Cache] 美股数据加载成功。")
    return df

def _create_map_from_df(df: pd.DataFrame, name_col: str, code_col: str) -> Dict[str, str]:
    """[辅助函数] 从DataFrame创建 {name: code} 字典的辅助函数"""
    df_unique = df.drop_duplicates(subset=[name_col])
    stock_map = pd.Series(
        df_unique[code_col].values, 
        index=df_unique[name_col]
    ).to_dict()
    return stock_map

def get_stock_map_by_market(
    market: str, 
    output_format: str = "dict" 
) -> Union[Dict[str, str], List[str], str]: 
    """
    根据市场 ('a', 'hk', 'us')，返回 {股票名称: 股票代码} 的字典，或所有股票代码的列表。
    
    Args:
        market (str): 市场代码，必须是 'a' (A股), 'hk' (港股), 或 'us' (美股)。
        output_format (str, optional): 
            返回的格式。默认为 "dict" ({名称: 代码})。
            - "dict": 返回 {股票名称: 股票代码} 的字典。
            - "code_list": 返回所有股票代码的列表 [代码1, 代码2, ...]。

    Returns:
        Union[Dict[str, str], List[str], str]: 
            - 成功 (dict): 返回 {股票名称: 股票代码} 的字典。
            - 成功 (code_list): 返回所有股票代码的列表。
            - 失败: 返回一个描述错误的字符串。
    """
    print(f"--- 正在为市场 '{market}' 获取股票地图 (格式: {output_format})... ---")
    VALID_FORMATS = ["dict", "code_list"]
    if output_format not in VALID_FORMATS:
        return f"错误：无效的 'output_format' 参数 '{output_format}'。有效选项为 {VALID_FORMATS}。"
    df_or_error: Union[pd.DataFrame, str]
    name_col, code_col = "", ""
    if market == 'a':
        df_or_error = _load_a_stock_data()
        name_col, code_col = '名称', '代码'
    elif market == 'hk':
        df_or_error = _load_hk_stock_data()
        name_col, code_col = '中文名称', '代码'
    elif market == 'us':
        df_or_error = _load_us_stock_data()
        name_col, code_col = 'name', 'symbol'
    else:
        return f"错误：无效的市场代码 '{market}'。有效代码为 'a', 'hk', 'us'。"
    if isinstance(df_or_error, str):
        return df_or_error 
    df = df_or_error 
    try:
        if output_format == "dict":
            return _create_map_from_df(df, name_col=name_col, code_col=code_col)
        elif output_format == "code_list":
            code_list = df[code_col].drop_duplicates().tolist()
            print(f"--- 市场 '{market}' 加载成功，返回 {len(code_list)} 个独特的代码。 ---")
            return code_list
    except KeyError as e:
        return f"错误：在处理DataFrame时缺少关键列: {e}。请检查CSV文件和列名定义。"
    except Exception as e:
        return f"错误：在格式化输出时发生未知错误: {e}"
    return "错误：未知的内部错误。"

def get_us_stock_map_by_exchange(
    exchange_name: str,
    output_format: str = "dict" 
) -> Union[Dict[str, str], List[str], str]: 
    """
    返回指定美股市场（例如 'NASDAQ', 'NYSE'）的所有股票 {name: symbol} 字典
    或所有股票代码 (symbol) 的列表。
    
    Args:
        exchange_name (str): 交易所的准确名称 (区分大小写)。
        output_format (str, optional): 
            返回的格式。默认为 "dict" ({name: symbol})。
            - "dict": 返回 {name: symbol} 的字典。
            - "code_list": 返回所有股票代码 (symbol) 的列表 [代码1, 代码2, ...]。

    Returns:
        Union[Dict[str, str], List[str], str]: 
            - 成功 (dict): 返回 {name: symbol} 的字典。
            - 成功 (code_list): 返回所有股票代码的列表。
            - 失败: 返回一个描述错误的字符串。
    """
    print(f"--- 正在为美股交易所 '{exchange_name}' 获取股票地图 (格式: {output_format})... ---")
    VALID_FORMATS = ["dict", "code_list"]
    if output_format not in VALID_FORMATS:
        return f"错误：无效的 'output_format' 参数 '{output_format}'。有效选项为 {VALID_FORMATS}。"
    df_or_error = _load_us_stock_data()
    if isinstance(df_or_error, str):
        return df_or_error 
    df = df_or_error
    if 'market' not in df.columns:
        return "错误: 美股CSV中缺少 'market' 列, 无法按交易所筛选。"
    try:
        filtered_df = df[df['market'] == exchange_name]
        if filtered_df.empty:
            available_markets = df['market'].unique().tolist()
            return (f"错误：在美股数据中未找到交易所 '{exchange_name}'。\n"
                    f"    - (请注意：此筛选区分大小写)\n"
                    f"    - 可用的市场示例: {available_markets[:10]}...")
        print(f"--- 筛选到 {len(filtered_df)} 只 '{exchange_name}' 市场的股票。 ---")
        if output_format == "dict":
            return _create_map_from_df(filtered_df, name_col='name', code_col='symbol')
        elif output_format == "code_list":
            code_list = filtered_df['symbol'].drop_duplicates().tolist()
            print(f"--- 市场 '{exchange_name}' 加载成功，返回 {len(code_list)} 个独特的代码。 ---")
            return code_list
    except KeyError as e:
        return f"错误：在处理DataFrame时缺少关键列 (name, symbol): {e}。"
    except Exception as e:
        return f"错误: 筛选交易所 '{exchange_name}' 时出错: {e}"
    return "错误：未知的内部错误。"


def get_stock_board_industry_list() -> Union[pd.DataFrame, str]:
    """
    获取东方财富-行业板块的所有板块列表
    
    Returns:
        Union[pd.DataFrame, str]: 
            - 成功: 返回包含板块代码和名称的DataFrame
            - 失败: 返回错误信息字符串
    """
    print("--- 正在获取东方财富行业板块列表... ---")
    try:
        df = ak.stock_board_industry_name_em()
        if df is None or df.empty:
            return "错误: 未能获取板块列表数据。"
        print(f"--- 成功获取 {len(df)} 个行业板块 ---")
        return df
    except Exception as e:
        error_msg = f"错误: 获取行业板块列表时出错: {str(e)}"
        print(f"  -> {error_msg}")
        return error_msg


def get_stock_board_industry_cons(symbol: str) -> Union[pd.DataFrame, str]:
    """
    获取东方财富-行业板块的成份股数据
    
    Args:
        symbol (str): 板块名称（如"小金属"）或板块代码（如"BK1027"）
    
    Returns:
        Union[pd.DataFrame, str]: 
            - 成功: 返回包含板块成份股数据的DataFrame，包含：
                - 代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额
                - 振幅、最高、最低、今开、昨收、换手率、市盈率-动态、市净率
            - 失败: 返回错误信息字符串
    """
    print(f"--- 正在获取板块 '{symbol}' 的成份股数据... ---")
    try:
        df = ak.stock_board_industry_cons_em(symbol=symbol)
        if df is None or df.empty:
            return f"错误: 未能获取板块 '{symbol}' 的成份股数据（可能板块名称或代码不正确）。"
        
        # 确保列名标准化
        if '代码' in df.columns:
            df = df.rename(columns={'代码': 'code', '名称': 'name'})
        
        print(f"--- 成功获取板块 '{symbol}' 的 {len(df)} 只成份股 ---")
        return df
    except Exception as e:
        error_msg = f"错误: 获取板块 '{symbol}' 成份股数据时出错: {str(e)}"
        print(f"  -> {error_msg}")
        return error_msg


def get_sw_index_third_info() -> Union[pd.DataFrame, str]:
    """
    获取申万三级行业信息
    
    Returns:
        Union[pd.DataFrame, str]: 
            - 成功: 返回包含申万三级行业信息的DataFrame，包含：
                - 行业代码、行业名称、上级行业、成份个数
                - 静态市盈率、TTM(滚动)市盈率、市净率、静态股息率
            - 失败: 返回错误信息字符串
    """
    print("--- 正在获取申万三级行业信息... ---")
    try:
        df = ak.sw_index_third_info()
        if df is None or df.empty:
            return "错误: 未能获取申万三级行业信息。"
        print(f"--- 成功获取 {len(df)} 个申万三级行业 ---")
        return df
    except Exception as e:
        error_msg = f"错误: 获取申万三级行业信息时出错: {str(e)}"
        print(f"  -> {error_msg}")
        return error_msg


def get_sw_index_third_cons(symbol: str) -> Union[pd.DataFrame, str]:
    """
    获取申万三级行业的成份股数据
    
    Args:
        symbol (str): 行业代码（如"850111.SI"），可以通过 get_sw_index_third_info() 获取
    
    Returns:
        Union[pd.DataFrame, str]: 
            - 成功: 返回包含行业成份股数据的DataFrame，包含：
                - 股票代码、股票简称、纳入时间、申万1/2/3级分类
                - 价格、市盈率、市盈率ttm、市净率、股息率、市值
                - 归母净利润同比增长、营业收入同比增长等
            - 失败: 返回错误信息字符串
    """
    print(f"--- 正在获取申万三级行业 '{symbol}' 的成份股数据... ---")
    try:
        df = ak.sw_index_third_cons(symbol=symbol)
        if df is None or df.empty:
            return f"错误: 未能获取行业 '{symbol}' 的成份股数据（可能行业代码不正确）。"
        print(f"--- 成功获取行业 '{symbol}' 的 {len(df)} 只成份股 ---")
        return df
    except Exception as e:
        error_msg = f"错误: 获取行业 '{symbol}' 成份股数据时出错: {str(e)}"
        print(f"  -> {error_msg}")
        return error_msg
