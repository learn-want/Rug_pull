import vaex
import pandas as pd
from decimal import Decimal
from multiprocessing import get_context
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool
import threading
import warnings
warnings.filterwarnings("ignore")

WETH='0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'
USDC='0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'
WBTC='0x2260fac5e5542a773aa44fbcfedf7c193bc2c599'
USDT='0xdac17f958d2ee523a2206206994597c13d831ec7'
DAI='0x6b175474e89094c44da98b954eedeac495271d0f'

sync=vaex.open('/local/scratch/exported/Ethereum_token_txs_data_TY_23/uniswapv2/rugpull2408/ new_uni-v2-syncs_right.hdf5')
sync.rename('reserve0','amount0')
sync.rename('reserve1','amount1')
sync.rename('token_contract_address','pair_contract_address')

# swap=vaex.open('/local/scratch/exported/Ethereum_token_txs_data_TY_23/uniswapv2/new_uni-v2-swaps_right.hdf5')
# # mint=vaex.open('/local/scratch/exported/Ethereum_token_txs_data_TY_23/uniswapv2/uni-v2-mints_right.hdf5')
# # burn=vaex.open('/local/scratch/exported/Ethereum_token_txs_data_TY_23/uniswapv2/uni-v2-burns_right.hdf5')
# burn=pd.read_csv('/local/scratch/exported/Ethereum_token_txs_data_TY_23/uniswapv2/uni-v2-burns.csv')
# mint=pd.read_csv('/local/scratch/exported/Ethereum_token_txs_data_TY_23/uniswapv2/uni-v2-mints_right1.csv')

# pairs=pd.read_csv('/local/scratch/exported/Ethereum_token_txs_data_TY_23/uniswapv2/uni-v2-pairs_right.csv',low_memory=False)
pairs=pd.read_csv('/local/scratch/exported/Ethereum_token_txs_data_TY_23/uniswapv2/rugpull2408/pairs_to20362089.csv')
pairs['pair_contract_address1']=pairs['pair_contract_address'].str.lower()
print('read pairs done')

# def update_event(row):
#     if len(swap[swap['pair_contract_address']==row['pair_contract_address']][swap['block_number']==row['block_number']][swap['transactionIndex']==row['transactionIndex']][swap['log_index']==row['log_index']+1])>0:
#         return 'swap'
#     elif len(burn[burn['pair_contract_address']==row['pair_contract_address']][burn['block_number']==row['block_number']][burn['transactionIndex']==row['transactionIndex']][burn['log_index']==row['log_index']+1])>0:
#         return 'burn'
#     else:
#         return 'sync'

def rugpull_detect(pair_id):
    sync_pair=sync[sync['pair_contract_address']==pair_id.lower()].copy()
    sync_pair=sync_pair.to_pandas_df()
    df_total=sync_pair.sort_values(by=['block_number','transactionIndex','log_index'])
    pair_reserves=df_total
    pair_reserves['amount0'] = pair_reserves['amount0'].apply(lambda x: Decimal(x))
    pair_reserves['amount1'] = pair_reserves['amount1'].apply(lambda x: Decimal(x))
    pair_reserves['amount0_before_tx'] = pair_reserves['amount0'].shift(fill_value=Decimal('0'))
            # calculate amount0_after_tx
    pair_reserves['amount0_after_tx']=pair_reserves['amount0'].apply(lambda x: Decimal(x))

    pair_reserves['amount1_before_tx'] = pair_reserves['amount1'].shift(fill_value=Decimal('0'))
            # calculate amount1_after_tx
    pair_reserves['amount1_after_tx']=pair_reserves['amount1'].apply(lambda x: Decimal(x))
    pair_reserves = pair_reserves.drop_duplicates()
    pair_reserves=pair_reserves.reset_index().drop(['index'],axis=1)
    pair_reserves['change_rate0'] = pair_reserves['amount0_after_tx'] / pair_reserves['amount0_before_tx'].apply(
            lambda x: Decimal(10**(-10)) if x == 0 else x)
    pair_reserves['change_rate1'] = pair_reserves['amount1_after_tx'] / pair_reserves['amount1_before_tx'].apply(lambda x: Decimal(10**(-10)) if x == 0 else x)

    token0_symbol = pairs.loc[pairs['pair_contract_address1'] == pair_id, 'token0_symbol'].item()
    token1_symbol = pairs.loc[pairs['pair_contract_address1'] == pair_id, 'token1_symbol'].item()

    token0_address = pairs.loc[pairs['pair_contract_address1'] == pair_id, 'token0_address'].item().lower()
    token1_address=pairs.loc[pairs['pair_contract_address1']==pair_id,'token1_address'].item().lower()
    if (token0_address not in [WETH,WBTC,USDT,USDC,DAI]) and (token1_address in [WETH,WBTC,USDT,USDC,DAI]):
        flag=1
    elif (token0_address in [WETH,WBTC,USDT,USDC,DAI]) and (token1_address not in [WETH,WBTC,USDT,USDC,DAI]):
        flag=0
    elif (token0_address  in [WETH,WBTC,USDT,USDC,DAI]) and (token1_address in [WETH,WBTC,USDT,USDC,DAI]):
        flag=2
    else:
        flag=-1
    for threshold in (np.arange(0.01, 0.5, 0.01)):
        if flag==1:
            rugpull=pair_reserves[((pair_reserves['change_rate1'])<=threshold)] # add absolute value symbol because for some decimal very large token, sometimes accu_mount may be less than 0, but this value is very small
        elif flag==0:
            rugpull=pair_reserves[((pair_reserves['change_rate0'])<=threshold)]
        elif flag==2:
            rugpull=pair_reserves[((pair_reserves['change_rate0']<=threshold))|((pair_reserves['change_rate1']<=threshold))]
        else:# token 0 and token 1 are both in the list
            rugpull=pair_reserves[((pair_reserves['change_rate0']<=threshold))|((pair_reserves['change_rate1']<=threshold))]
        if len(rugpull)>0:
            # pairs.at[pairs['pair_contract_address1']==pair_id,'is_rug']=1
            #rugpull.loc[:, 'token0_symbol'] = token0_symbol
            rugpull['token0_symbol']=token0_symbol
            rugpull['token1_symbol']=token1_symbol
            rugpull['token0_address']=token0_address
            rugpull['token1_address']=token1_address
            rugpull['flag']=flag
            #rugpull['tx_number']取这一行的index
            rugpull['tx_number']=rugpull.index*2
            threshold=np.around(threshold,2)
            rugpull['threshold']=threshold
            # rugpull['event']=rugpull.apply(update_event,axis=1)
            with threading.Lock():
                # rugpull.to_csv(f'/local/scratch/exported/Uniswap_data_TY_23/rugpull_detect/rugpull_deetect_result_until202308/ground_truth_231223/0_2/rugpull_ground_truth_{threshold}.csv',mode='a',header=not os.path.exists(f'/local/scratch/exported/Uniswap_data_TY_23/rugpull_detect/rugpull_deetect_result_until202308/ground_truth_231223/0_2/rugpull_ground_truth_{threshold}.csv'),index=False)
                rugpull.to_csv(f'/local/scratch/exported/Uniswap_data_TY_23/rugpull_detect/rugpull_deetect_result_until2407/0_4/rugpull_ground_truth_{threshold}.csv',mode='a',header=not os.path.exists(f'/local/scratch/exported/Uniswap_data_TY_23/rugpull_detect/rugpull_deetect_result_until2407/0_4/rugpull_ground_truth_{threshold}.csv'),index=False)
                # print(f'{pair_id} has rugpull when threshold={threshold},end')
        else:
            pass
            threshold=np.around(threshold,2)
            # print(f"{pair_id} doesn't has rugpull when threshlod={threshold},end")
    else:
        # print(f"{pair_id} doesn't have any mint,swap,burn,end")
        pass  





# List of dynamic arguments
pairs_id = pairs['pair_contract_address1'].unique()
pairs_id = list(pairs_id)[:40000]
#将 pairs_id 分成 1000 份，每一份 200个 pair_id
gap=200
pairs_id_group=[pairs_id[i:i + gap] if len(pairs_id)-i >gap else pairs_id[i:] for i in range(0, len(pairs_id), gap)]

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
max_workers = 10
with tqdm(total=len(pairs_id_group), desc='Progress') as progress_bar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务给线程池
        for pair_ids in tqdm(pairs_id_group):
            futures = [executor.submit(rugpull_detect, pair) for pair in pair_ids] #age_thresholds 是一个值的列表
            # 等待所有任务完成
            for future, pair_id in zip(futures,pair_ids):  # Add pair_address
                from_address=future.result()
            progress_bar.update(1)
progress_bar.close()


            
            






