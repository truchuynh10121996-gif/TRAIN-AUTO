"""
Streamlit App: Synthetic Data Generator cho Autoencoder Fraud Detection
=========================================================================
Ứng dụng sinh dữ liệu synthetic chất lượng cao để huấn luyện mô hình Autoencoder
phát hiện gian lận (Hack/ATO) và lừa đảo (Scam) trong ngân hàng Việt Nam.

Author: AI Assistant
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import io
import json
import hashlib
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS - Hằng số cấu hình cho bối cảnh Việt Nam
# ============================================================================

# Danh sách ngân hàng Việt Nam
VN_BANKS = ['VCB', 'TCB', 'MB', 'BIDV', 'VPB', 'ACB', 'TPB', 'STB',
            'SHB', 'MSB', 'VIB', 'OCB', 'HDBank', 'LPB', 'Eximbank']

# Các mức thu nhập (triệu VND/tháng)
INCOME_LEVELS = {
    1: {'name': 'Thấp', 'range': (5_000_000, 15_000_000), 'balance_range': (2_000_000, 30_000_000)},
    2: {'name': 'Trung bình', 'range': (15_000_000, 30_000_000), 'balance_range': (10_000_000, 100_000_000)},
    3: {'name': 'Khá', 'range': (30_000_000, 70_000_000), 'balance_range': (50_000_000, 500_000_000)},
    4: {'name': 'Cao', 'range': (70_000_000, 200_000_000), 'balance_range': (200_000_000, 2_000_000_000)}
}

# Số tiền chẵn phổ biến (VND)
ROUND_AMOUNTS = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000,
                 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000]

# Ngày lễ Việt Nam (dạng (tháng, ngày))
VN_HOLIDAYS = [
    (1, 1),   # Tết Dương lịch
    (4, 30),  # Giải phóng miền Nam
    (5, 1),   # Quốc tế Lao động
    (9, 2),   # Quốc khánh
    (4, 10),  # Giỗ tổ Hùng Vương (ước lượng)
]

# Tỉnh/Thành phố Việt Nam
VN_PROVINCES = ['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Hải Phòng', 'Cần Thơ',
                'Bình Dương', 'Đồng Nai', 'Bắc Ninh', 'Quảng Ninh', 'Thanh Hóa',
                'Nghệ An', 'Hà Tĩnh', 'Khánh Hòa', 'Bà Rịa-VT', 'Long An']

# Feature columns (30 features theo yêu cầu)
FEATURE_COLUMNS = [
    # Amount (5)
    'amount_log', 'amount_to_balance_ratio', 'z_score_amount',
    'is_round_amount', 'amount_std_7d',
    # Temporal (6)
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'is_night_transaction', 'is_salary_period',
    # Velocity (5)
    'tx_count_1h', 'tx_count_24h', 'minutes_since_last_tx',
    'velocity_change', 'amount_acceleration',
    # Recipient (6)
    'is_new_recipient', 'recipient_tx_count', 'is_same_bank',
    'tx_count_to_same_recipient_24h', 'recipient_account_age_days',
    'time_since_last_tx_to_recipient',
    # Device/Channel (3)
    'is_new_device', 'channel_encoded', 'is_usual_location',
    # Account (2)
    'account_age_days', 'income_level',
    # Scam-specific (3)
    'is_incremental_amount', 'is_during_business_hours', 'recipient_total_received_24h'
]


# ============================================================================
# HELPER FUNCTIONS - Các hàm tiện ích
# ============================================================================

def set_seed(seed: int) -> None:
    """Thiết lập random seed để đảm bảo reproducibility"""
    np.random.seed(seed)


def generate_transaction_id() -> str:
    """Sinh mã giao dịch unique"""
    return hashlib.md5(str(np.random.random()).encode()).hexdigest()[:16].upper()


def generate_account_number(bank: str) -> str:
    """Sinh số tài khoản theo format ngân hàng VN"""
    return f"{bank}_{np.random.randint(1000000000, 9999999999)}"


def generate_device_id() -> str:
    """Sinh device ID"""
    return hashlib.md5(str(np.random.random()).encode()).hexdigest()[:12].upper()


def is_round_amount_check(amount: float) -> int:
    """Kiểm tra số tiền có phải số chẵn đẹp không"""
    return 1 if amount % 50000 == 0 else 0


def get_round_amount(income_level: int, is_large: bool = False) -> float:
    """Sinh số tiền chẵn dựa trên income level"""
    if income_level == 1:
        amounts = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]
    elif income_level == 2:
        amounts = [200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
    elif income_level == 3:
        amounts = [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
    else:
        amounts = [1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000]

    if is_large:
        amounts = amounts[len(amounts)//2:]
    return np.random.choice(amounts)


def generate_timestamp_in_range(start_date: datetime, end_date: datetime,
                                 hour_weights: Optional[List[float]] = None) -> datetime:
    """Sinh timestamp trong khoảng thời gian với trọng số giờ"""
    # Random ngày
    delta_days = (end_date - start_date).days
    random_days = np.random.randint(0, max(delta_days, 1))
    date = start_date + timedelta(days=random_days)

    # Random giờ với trọng số (giờ cao điểm)
    if hour_weights is None:
        # Trọng số mặc định: cao điểm 8-11h, 14-16h, 19:30-21:30
        hour_weights = [0.5, 0.3, 0.2, 0.2, 0.2, 0.3,  # 0-5h
                       0.5, 0.8, 1.5, 1.5, 1.5, 1.2,   # 6-11h
                       0.8, 0.8, 1.3, 1.3, 1.2, 0.9,   # 12-17h
                       0.8, 1.2, 1.5, 1.3, 0.8, 0.5]   # 18-23h

    hour_weights = np.array(hour_weights) / sum(hour_weights)
    hour = np.random.choice(24, p=hour_weights)
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)

    return date.replace(hour=hour, minute=minute, second=second)


def generate_night_timestamp(date: datetime) -> datetime:
    """Sinh timestamp vào ban đêm (23h-4h)"""
    hour = np.random.choice([23, 0, 1, 2, 3, 4])
    return date.replace(hour=hour, minute=np.random.randint(0, 60),
                       second=np.random.randint(0, 60))


def generate_business_hours_timestamp(date: datetime) -> datetime:
    """Sinh timestamp trong giờ hành chính (8h-17h, thứ 2-6)"""
    # Đảm bảo là ngày trong tuần (0=Mon, 6=Sun)
    while date.weekday() >= 5:
        date = date + timedelta(days=1)

    hour = np.random.randint(8, 17)
    return date.replace(hour=hour, minute=np.random.randint(0, 60),
                       second=np.random.randint(0, 60))


# ============================================================================
# USER & RECIPIENT GENERATION
# ============================================================================

def generate_users(n_users: int, start_date: datetime, seed: int = 42) -> pd.DataFrame:
    """
    Sinh thông tin người dùng

    Returns:
        DataFrame với columns: user_id, bank, income_level, account_created_date,
                              balance, primary_device, usual_province
    """
    set_seed(seed)

    users = []
    for i in range(n_users):
        # Phân phối income level theo thực tế VN
        income_level = np.random.choice([1, 2, 3, 4], p=[0.35, 0.40, 0.18, 0.07])
        income_info = INCOME_LEVELS[income_level]

        # Ngày tạo tài khoản (từ 1 tháng đến 5 năm trước start_date)
        days_before = np.random.randint(30, 5*365)
        account_created = start_date - timedelta(days=days_before)

        # Số dư trung bình
        balance = np.random.uniform(*income_info['balance_range'])

        # Số giao dịch trung bình mỗi ngày (dựa theo income)
        avg_tx_per_day = np.random.uniform(0.5, 2.0) * (income_level * 0.5 + 0.5)

        users.append({
            'user_id': f'U{i:06d}',
            'bank': np.random.choice(VN_BANKS),
            'income_level': income_level,
            'account_created_date': account_created,
            'balance': balance,
            'primary_device': generate_device_id(),
            'usual_province': np.random.choice(VN_PROVINCES[:5], p=[0.35, 0.35, 0.10, 0.10, 0.10]),
            'avg_tx_per_day': avg_tx_per_day
        })

    return pd.DataFrame(users)


def generate_recipients(n_recipients: int, start_date: datetime, seed: int = 42) -> pd.DataFrame:
    """
    Sinh thông tin người nhận tiền

    Returns:
        DataFrame với columns: recipient_id, bank, account_created_date, is_money_mule
    """
    set_seed(seed + 1)

    recipients = []
    for i in range(n_recipients):
        # 5% là money mule (tài khoản mới, thường dùng cho fraud)
        is_money_mule = np.random.random() < 0.05

        if is_money_mule:
            # Money mule: tài khoản mới (<30 ngày)
            days_before = np.random.randint(1, 30)
        else:
            # Tài khoản bình thường
            days_before = np.random.randint(30, 3*365)

        account_created = start_date - timedelta(days=days_before)

        recipients.append({
            'recipient_id': f'R{i:06d}',
            'bank': np.random.choice(VN_BANKS),
            'account_created_date': account_created,
            'is_money_mule': is_money_mule
        })

    return pd.DataFrame(recipients)


# ============================================================================
# NORMAL TRANSACTION GENERATION
# ============================================================================

def generate_normal_transactions(users_df: pd.DataFrame,
                                  recipients_df: pd.DataFrame,
                                  n_transactions: int,
                                  start_date: datetime,
                                  end_date: datetime,
                                  seed: int = 42) -> pd.DataFrame:
    """
    Sinh giao dịch bình thường với pattern thực tế

    Pattern normal:
    - Device quen thuộc (>90%)
    - Location quen thuộc (>85%)
    - Giờ bình thường (chủ yếu 7h-22h)
    - Số tiền theo income level
    - Có xu hướng giao dịch với người quen
    """
    set_seed(seed)

    transactions = []
    n_users = len(users_df)
    n_recipients = len(recipients_df)

    # Tạo mapping user -> danh sách recipient quen thuộc
    user_frequent_recipients = {}
    for _, user in users_df.iterrows():
        # Mỗi user có 3-10 recipient quen thuộc
        n_frequent = np.random.randint(3, 11)
        user_frequent_recipients[user['user_id']] = np.random.choice(
            recipients_df['recipient_id'].values, n_frequent, replace=False
        ).tolist()

    for _ in range(n_transactions):
        # Random user
        user = users_df.iloc[np.random.randint(n_users)]
        user_id = user['user_id']
        income_level = user['income_level']

        # Timestamp với pattern giờ cao điểm
        timestamp = generate_timestamp_in_range(start_date, end_date)

        # Amount theo income level (70% số chẵn, 30% số lẻ)
        if np.random.random() < 0.7:
            amount = get_round_amount(income_level)
        else:
            income_info = INCOME_LEVELS[income_level]
            max_tx = min(user['balance'] * 0.3, income_info['range'][1] * 0.5)
            amount = np.random.uniform(10000, max_tx)
            amount = round(amount / 1000) * 1000  # Làm tròn 1000

        # Recipient: 70% người quen, 30% người mới
        if np.random.random() < 0.7 and user_id in user_frequent_recipients:
            recipient_id = np.random.choice(user_frequent_recipients[user_id])
        else:
            recipient_id = recipients_df.iloc[np.random.randint(n_recipients)]['recipient_id']

        recipient = recipients_df[recipients_df['recipient_id'] == recipient_id].iloc[0]

        # Device: 95% device chính, 5% device khác
        is_new_device = 1 if np.random.random() < 0.05 else 0
        device_id = generate_device_id() if is_new_device else user['primary_device']

        # Channel: Mobile 60%, Web 25%, ATM 10%, POS 5%
        channel = np.random.choice([0, 1, 2, 3], p=[0.60, 0.25, 0.10, 0.05])

        # Location: 90% quen thuộc
        is_usual_location = 1 if np.random.random() < 0.90 else 0
        province = user['usual_province'] if is_usual_location else np.random.choice(VN_PROVINCES)

        transactions.append({
            'transaction_id': generate_transaction_id(),
            'user_id': user_id,
            'recipient_id': recipient_id,
            'timestamp': timestamp,
            'amount': amount,
            'balance_before': user['balance'],
            'device_id': device_id,
            'channel': channel,
            'province': province,
            'user_bank': user['bank'],
            'recipient_bank': recipient['bank'],
            'user_account_created': user['account_created_date'],
            'recipient_account_created': recipient['account_created_date'],
            'income_level': income_level,
            'is_fraud': 0,
            'fraud_type': 'normal',
            'is_new_device_raw': is_new_device,
            'is_usual_location_raw': is_usual_location
        })

    return pd.DataFrame(transactions)


# ============================================================================
# HACK/ATO FRAUD GENERATION
# ============================================================================

def generate_hack_transactions(users_df: pd.DataFrame,
                               recipients_df: pd.DataFrame,
                               n_fraud_cases: int,
                               start_date: datetime,
                               end_date: datetime,
                               seed: int = 42) -> pd.DataFrame:
    """
    Sinh giao dịch gian lận loại Hack/Account Takeover

    Scenarios:
    a) SIM Swap Attack
    b) Credential Stuffing
    c) Malware/Keylogger
    d) Session Hijacking

    Pattern đặc trưng:
    - Device mới, IP/location lạ
    - Đêm khuya (23h-4h)
    - Nhiều giao dịch liên tiếp (<5 phút)
    - Rút gần hết số dư
    - Chuyển cho tài khoản money mule
    """
    set_seed(seed + 100)

    transactions = []
    n_users = len(users_df)

    # Lọc money mule recipients
    money_mules = recipients_df[recipients_df['is_money_mule'] == True]['recipient_id'].tolist()
    if len(money_mules) < 10:
        # Nếu không đủ, thêm random recipients mới tạo
        money_mules.extend(recipients_df.nsmallest(50, 'account_created_date')['recipient_id'].tolist())

    for case_idx in range(n_fraud_cases):
        # Chọn scenario
        scenario = np.random.choice(['sim_swap', 'credential', 'malware', 'session_hijack'],
                                   p=[0.25, 0.35, 0.25, 0.15])

        # Random user bị hack
        user = users_df.iloc[np.random.randint(n_users)]

        # Thời điểm tấn công (chủ yếu đêm khuya)
        attack_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))

        # Số giao dịch trong 1 đợt tấn công: 2-5 giao dịch
        n_tx_per_attack = np.random.randint(2, 6)

        # Số dư còn lại sau tấn công
        remaining_balance = user['balance']

        for tx_idx in range(n_tx_per_attack):
            # Timestamp: đêm khuya, liên tiếp nhau 1-5 phút
            if tx_idx == 0:
                timestamp = generate_night_timestamp(attack_date)
            else:
                # Giao dịch tiếp theo: 1-5 phút sau
                timestamp = transactions[-1]['timestamp'] + timedelta(minutes=np.random.randint(1, 6))

            # Amount: rút nhiều, giao dịch cuối rút gần hết
            if tx_idx < n_tx_per_attack - 1:
                # Rút 20-40% mỗi lần
                amount = remaining_balance * np.random.uniform(0.2, 0.4)
            else:
                # Lần cuối: rút 70-95% số còn lại
                amount = remaining_balance * np.random.uniform(0.7, 0.95)

            # Làm tròn số tiền
            amount = round(amount / 100000) * 100000
            amount = max(amount, 500000)  # Tối thiểu 500k

            remaining_balance -= amount

            # Recipient: money mule hoặc tài khoản mới
            recipient_id = np.random.choice(money_mules)
            recipient = recipients_df[recipients_df['recipient_id'] == recipient_id].iloc[0]

            # Device: luôn mới
            device_id = generate_device_id()

            # Channel: chủ yếu Mobile app
            channel = np.random.choice([0, 1], p=[0.7, 0.3])

            # Location: lạ
            province = np.random.choice(VN_PROVINCES[5:])  # Tỉnh khác

            transactions.append({
                'transaction_id': generate_transaction_id(),
                'user_id': user['user_id'],
                'recipient_id': recipient_id,
                'timestamp': timestamp,
                'amount': amount,
                'balance_before': remaining_balance + amount,
                'device_id': device_id,
                'channel': channel,
                'province': province,
                'user_bank': user['bank'],
                'recipient_bank': recipient['bank'],
                'user_account_created': user['account_created_date'],
                'recipient_account_created': recipient['account_created_date'],
                'income_level': user['income_level'],
                'is_fraud': 1,
                'fraud_type': f'hack_{scenario}',
                'is_new_device_raw': 1,
                'is_usual_location_raw': 0
            })

    return pd.DataFrame(transactions)


# ============================================================================
# SCAM FRAUD GENERATION
# ============================================================================

def generate_scam_transactions(users_df: pd.DataFrame,
                               recipients_df: pd.DataFrame,
                               n_fraud_cases: int,
                               start_date: datetime,
                               end_date: datetime,
                               seed: int = 42) -> pd.DataFrame:
    """
    Sinh giao dịch lừa đảo (Scam)

    Scenarios tại VN:
    a) Giả danh công an/viện kiểm sát
    b) Lừa đầu tư (Crypto/Forex)
    c) Lừa tình cảm
    d) Giả danh người thân
    e) Lừa trúng thưởng
    f) Lừa mua hàng online
    g) Việc làm tại nhà

    Pattern đặc trưng:
    - Device quen thuộc (chính chủ thực hiện)
    - Giờ hành chính
    - Số tiền tăng dần
    - Chuyển nhiều lần cho cùng người
    - Số tiền chẵn đẹp
    """
    set_seed(seed + 200)

    transactions = []
    n_users = len(users_df)
    n_recipients = len(recipients_df)

    scenarios = ['police_scam', 'investment_scam', 'romance_scam',
                 'family_scam', 'prize_scam', 'shopping_scam', 'job_scam']
    scenario_probs = [0.25, 0.30, 0.10, 0.10, 0.08, 0.10, 0.07]

    for case_idx in range(n_fraud_cases):
        # Chọn scenario
        scenario = np.random.choice(scenarios, p=scenario_probs)

        # Random user bị lừa
        user = users_df.iloc[np.random.randint(n_users)]

        # Random scammer (recipient cố định cho mỗi case)
        scammer_id = recipients_df.iloc[np.random.randint(n_recipients)]['recipient_id']
        scammer = recipients_df[recipients_df['recipient_id'] == scammer_id].iloc[0]

        # Thời điểm bắt đầu bị lừa
        scam_start = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days - 7))

        # Số lần chuyển tiền: 2-8 lần tùy scenario
        if scenario in ['investment_scam', 'job_scam']:
            n_transfers = np.random.randint(3, 9)  # Nhiều lần hơn
        elif scenario in ['police_scam', 'family_scam']:
            n_transfers = np.random.randint(1, 4)  # Ít lần, số tiền lớn
        else:
            n_transfers = np.random.randint(2, 6)

        # Số tiền cơ bản dựa theo income level
        base_amount = get_round_amount(user['income_level'], is_large=True)

        for tx_idx in range(n_transfers):
            # Timestamp: giờ hành chính, cách nhau 1-3 ngày
            if tx_idx == 0:
                timestamp = generate_business_hours_timestamp(scam_start)
            else:
                days_gap = np.random.randint(0, 4)  # 0-3 ngày
                new_date = transactions[-1]['timestamp'] + timedelta(days=days_gap)
                timestamp = generate_business_hours_timestamp(new_date)

            # Amount: tăng dần (đặc trưng scam)
            if scenario in ['investment_scam', 'job_scam']:
                # Tăng dần theo cấp số nhân
                multiplier = 1.5 ** tx_idx
                amount = base_amount * multiplier
            elif scenario == 'police_scam':
                # Số tiền lớn ngay từ đầu
                amount = user['balance'] * np.random.uniform(0.3, 0.5)
            else:
                # Tăng dần tuyến tính
                amount = base_amount * (1 + tx_idx * 0.3)

            # Làm tròn số đẹp
            if amount < 1_000_000:
                amount = round(amount / 50000) * 50000
            elif amount < 10_000_000:
                amount = round(amount / 100000) * 100000
            else:
                amount = round(amount / 1000000) * 1000000

            amount = max(amount, 100000)  # Tối thiểu 100k
            amount = min(amount, user['balance'] * 0.9)  # Không quá số dư

            # Device: chính chủ (device quen thuộc)
            device_id = user['primary_device']

            # Channel: Mobile app là chủ yếu
            channel = np.random.choice([0, 1], p=[0.8, 0.2])

            # Location: quen thuộc (chính chủ làm)
            province = user['usual_province']

            transactions.append({
                'transaction_id': generate_transaction_id(),
                'user_id': user['user_id'],
                'recipient_id': scammer_id,
                'timestamp': timestamp,
                'amount': amount,
                'balance_before': user['balance'],
                'device_id': device_id,
                'channel': channel,
                'province': province,
                'user_bank': user['bank'],
                'recipient_bank': scammer['bank'],
                'user_account_created': user['account_created_date'],
                'recipient_account_created': scammer['account_created_date'],
                'income_level': user['income_level'],
                'is_fraud': 1,
                'fraud_type': f'scam_{scenario}',
                'is_new_device_raw': 0,
                'is_usual_location_raw': 1
            })

    return pd.DataFrame(transactions)


# ============================================================================
# FEATURE ENGINEERING - Tính 30 features
# ============================================================================

def compute_features(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    Tính toán đầy đủ 30 features cho từng giao dịch

    QUAN TRỌNG: Tất cả rolling features đều dùng shift(1) để tránh data leakage

    Args:
        df: DataFrame với các giao dịch đã sort theo timestamp
        progress_callback: Hàm callback để cập nhật progress

    Returns:
        DataFrame với đầy đủ 30 features
    """
    # Sort theo timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    total_steps = 10
    current_step = 0

    def update_progress(step_name):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps, step_name)

    # === AMOUNT FEATURES (5) ===
    update_progress("Tính Amount features...")

    # 1. amount_log
    df['amount_log'] = np.log1p(df['amount'])

    # 2. amount_to_balance_ratio
    df['amount_to_balance_ratio'] = df['amount'] / df['balance_before'].clip(lower=1)

    # 3. z_score_amount (rolling 30 ngày, shift(1) để tránh leak)
    df['amount_mean_30d'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    )
    df['amount_std_30d'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).std()
    )
    df['amount_std_30d'] = df['amount_std_30d'].fillna(df['amount_mean_30d'] * 0.5).clip(lower=1)
    df['z_score_amount'] = (df['amount'] - df['amount_mean_30d']) / df['amount_std_30d']
    df['z_score_amount'] = df['z_score_amount'].fillna(0).clip(-5, 10)

    # 4. is_round_amount
    df['is_round_amount'] = (df['amount'] % 50000 == 0).astype(int)

    # 5. amount_std_7d (shift(1))
    df['amount_std_7d'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
    )
    df['amount_std_7d'] = df['amount_std_7d'].fillna(0)

    # === TEMPORAL FEATURES (6) ===
    update_progress("Tính Temporal features...")

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day

    # 6. hour_sin
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)

    # 7. hour_cos
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 8. day_of_week_sin
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)

    # 9. day_of_week_cos
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # 10. is_night_transaction (23h-5h)
    df['is_night_transaction'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)

    # 11. is_salary_period (ngày 25-5 hàng tháng)
    df['is_salary_period'] = ((df['day_of_month'] >= 25) | (df['day_of_month'] <= 5)).astype(int)

    # === VELOCITY FEATURES (5) ===
    update_progress("Tính Velocity features...")

    df['timestamp_unix'] = df['timestamp'].astype(np.int64) // 10**9

    # 12. tx_count_1h (số giao dịch trong 1 giờ qua)
    # Tính bằng cách đếm số giao dịch trong 1 giờ trước đó
    def calc_rolling_count(group, hours):
        """Tính số giao dịch trong N giờ trước (không tính giao dịch hiện tại)"""
        group = group.sort_values('timestamp')
        timestamps = group['timestamp'].values
        counts = []
        for i, ts in enumerate(timestamps):
            # Đếm số giao dịch trong khoảng [ts - N hours, ts) - không tính giao dịch hiện tại
            cutoff = ts - np.timedelta64(hours, 'h')
            count = np.sum((timestamps[:i] >= cutoff) & (timestamps[:i] < ts))
            counts.append(count)
        return pd.Series(counts, index=group.index)

    df['tx_count_1h'] = df.groupby('user_id', group_keys=False).apply(
        lambda x: calc_rolling_count(x, 1)
    ).fillna(0).astype(int)

    # 13. tx_count_24h
    df['tx_count_24h'] = df.groupby('user_id', group_keys=False).apply(
        lambda x: calc_rolling_count(x, 24)
    ).fillna(0).astype(int)

    # 14. minutes_since_last_tx
    df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
    df['minutes_since_last_tx'] = (
        (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 60
    ).fillna(10000).clip(upper=10000)

    # 15. velocity_change (tx_count_24h / avg_tx_24h_30d)
    df['avg_tx_24h_30d'] = df.groupby('user_id')['tx_count_24h'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    ).fillna(1).clip(lower=0.1)
    df['velocity_change'] = df['tx_count_24h'] / df['avg_tx_24h_30d']
    df['velocity_change'] = df['velocity_change'].fillna(1).clip(upper=10)

    # 16. amount_acceleration (tốc độ tăng số tiền)
    df['prev_amount'] = df.groupby('user_id')['amount'].shift(1)
    df['amount_acceleration'] = ((df['amount'] - df['prev_amount']) / df['prev_amount'].clip(lower=1))
    df['amount_acceleration'] = df['amount_acceleration'].fillna(0).clip(-2, 5)

    # === RECIPIENT FEATURES (6) ===
    update_progress("Tính Recipient features...")

    # Tính recipient_tx_count theo thứ tự thời gian
    df['user_recipient_key'] = df['user_id'] + '_' + df['recipient_id']

    # 17. recipient_tx_count (số lần đã giao dịch với người này TRƯỚC đó)
    df['recipient_tx_count'] = df.groupby('user_recipient_key').cumcount()

    # 18. is_new_recipient
    df['is_new_recipient'] = (df['recipient_tx_count'] == 0).astype(int)

    # 19. is_same_bank
    df['is_same_bank'] = (df['user_bank'] == df['recipient_bank']).astype(int)

    # 20. tx_count_to_same_recipient_24h
    df['tx_count_to_same_recipient_24h'] = df.groupby('user_recipient_key', group_keys=False).apply(
        lambda x: calc_rolling_count(x, 24)
    ).fillna(0).astype(int)

    # 21. recipient_account_age_days
    df['recipient_account_age_days'] = (
        df['timestamp'] - pd.to_datetime(df['recipient_account_created'])
    ).dt.days.clip(lower=0)

    # 22. time_since_last_tx_to_recipient (ngày)
    df['prev_tx_to_recipient'] = df.groupby('user_recipient_key')['timestamp'].shift(1)
    df['time_since_last_tx_to_recipient'] = (
        (df['timestamp'] - df['prev_tx_to_recipient']).dt.days
    ).fillna(9999).clip(upper=9999)

    # === DEVICE/CHANNEL FEATURES (3) ===
    update_progress("Tính Device/Channel features...")

    # 23. is_new_device (dùng raw value đã tính khi sinh data)
    df['is_new_device'] = df['is_new_device_raw']

    # 24. channel_encoded (đã có sẵn)
    df['channel_encoded'] = df['channel']

    # 25. is_usual_location (dùng raw value)
    df['is_usual_location'] = df['is_usual_location_raw']

    # === ACCOUNT FEATURES (2) ===
    update_progress("Tính Account features...")

    # 26. account_age_days
    df['account_age_days'] = (
        df['timestamp'] - pd.to_datetime(df['user_account_created'])
    ).dt.days.clip(lower=0)

    # 27. income_level (đã có sẵn)
    # df['income_level'] đã có

    # === SCAM-SPECIFIC FEATURES (3) ===
    update_progress("Tính Scam-specific features...")

    # 28. is_incremental_amount (số tiền tăng so với lần trước cùng recipient)
    df['prev_amount_to_recipient'] = df.groupby('user_recipient_key')['amount'].shift(1)
    df['is_incremental_amount'] = (df['amount'] > df['prev_amount_to_recipient']).fillna(False).astype(int)
    df = df.drop(columns=['prev_amount_to_recipient'], errors='ignore')

    # 29. is_during_business_hours (8h-17h, thứ 2-6)
    df['is_during_business_hours'] = (
        (df['hour'] >= 8) & (df['hour'] <= 17) & (df['day_of_week'] < 5)
    ).astype(int)

    # 30. recipient_total_received_24h (chỉ tính từ giao dịch TRƯỚC)
    # Đây là tổng tiền recipient nhận được trong 24h từ TẤT CẢ các nguồn
    def calc_rolling_sum(group, hours):
        """Tính tổng amount trong N giờ trước (không tính giao dịch hiện tại)"""
        group = group.sort_values('timestamp')
        timestamps = group['timestamp'].values
        amounts = group['amount'].values
        sums = []
        for i, ts in enumerate(timestamps):
            cutoff = ts - np.timedelta64(hours, 'h')
            mask = (timestamps[:i] >= cutoff) & (timestamps[:i] < ts)
            total = np.sum(amounts[:i][mask])
            sums.append(total)
        return pd.Series(sums, index=group.index)

    df['recipient_total_received_24h'] = df.groupby('recipient_id', group_keys=False).apply(
        lambda x: calc_rolling_sum(x, 24)
    ).fillna(0)

    update_progress("Chuẩn hóa features...")

    # Xóa các cột tạm
    cols_to_drop = ['hour', 'day_of_week', 'day_of_month', 'timestamp_unix',
                    'amount_mean_30d', 'amount_std_30d', 'prev_timestamp',
                    'avg_tx_24h_30d', 'prev_amount', 'user_recipient_key',
                    'prev_tx_to_recipient', 'is_new_device_raw', 'is_usual_location_raw']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    update_progress("Hoàn thành feature engineering!")

    return df


# ============================================================================
# TIME-BASED SPLIT - Chia dữ liệu theo thời gian
# ============================================================================

def time_based_split(df: pd.DataFrame,
                     train_ratio: float = 0.67,
                     val_ratio: float = 0.16,
                     test_ratio: float = 0.17) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia dữ liệu theo thời gian để tránh data leakage

    Args:
        df: DataFrame đã sort theo timestamp
        train_ratio: Tỷ lệ train (mặc định 67%)
        val_ratio: Tỷ lệ validation (mặc định 16%)
        test_ratio: Tỷ lệ test (mặc định 17%)

    Returns:
        train_df, val_df, test_df

    QUAN TRỌNG:
    - Train set CHỈ chứa giao dịch normal (is_fraud = 0)
    - Validation và Test chứa cả normal + fraud
    """
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Tính cutoff points dựa trên số lượng giao dịch
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split
    train_all = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Train set: CHỈ lấy normal transactions
    train_df = train_all[train_all['is_fraud'] == 0].copy()

    # Thêm cột split để đánh dấu
    train_df['split'] = 'train'
    val_df['split'] = 'validation'
    test_df['split'] = 'test'

    return train_df, val_df, test_df


# ============================================================================
# NORMALIZATION - Chuẩn hóa dữ liệu
# ============================================================================

def fit_and_transform_scaler(train_df: pd.DataFrame,
                             val_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit scaler trên TRAIN set và transform tất cả các set

    QUAN TRỌNG: Scaler chỉ được fit trên train để tránh data leakage
    """
    scaler = StandardScaler()

    # Fit chỉ trên train
    scaler.fit(train_df[feature_cols])

    # Transform tất cả
    train_normalized = train_df.copy()
    val_normalized = val_df.copy()
    test_normalized = test_df.copy()

    train_normalized[feature_cols] = scaler.transform(train_df[feature_cols])
    val_normalized[feature_cols] = scaler.transform(val_df[feature_cols])
    test_normalized[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_normalized, val_normalized, test_normalized, scaler


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def create_data_dictionary() -> pd.DataFrame:
    """Tạo data dictionary cho 30 features"""
    dictionary = [
        # Amount (5)
        {'feature_name': 'amount_log', 'data_type': 'float',
         'description': 'Log của số tiền giao dịch', 'calculation': 'log(amount + 1)',
         'range': '[0, 20]', 'notes': 'Log base e'},
        {'feature_name': 'amount_to_balance_ratio', 'data_type': 'float',
         'description': 'Tỷ lệ giao dịch/số dư hiện tại', 'calculation': 'amount / balance_before',
         'range': '[0, 1+]', 'notes': 'Có thể > 1 nếu overdraft'},
        {'feature_name': 'z_score_amount', 'data_type': 'float',
         'description': 'Z-score của amount so với 30 ngày', 'calculation': '(amount - mean_30d) / std_30d',
         'range': '[-5, 10+]', 'notes': 'Clipped, dùng shift(1)'},
        {'feature_name': 'is_round_amount', 'data_type': 'int',
         'description': 'Số tiền chia hết 50,000', 'calculation': 'amount % 50000 == 0',
         'range': '[0, 1]', 'notes': 'Binary'},
        {'feature_name': 'amount_std_7d', 'data_type': 'float',
         'description': 'Độ lệch chuẩn số tiền 7 ngày', 'calculation': 'rolling std 7 ngày',
         'range': '[0, inf]', 'notes': 'Dùng shift(1)'},

        # Temporal (6)
        {'feature_name': 'hour_sin', 'data_type': 'float',
         'description': 'Sine của giờ', 'calculation': 'sin(2π × hour/24)',
         'range': '[-1, 1]', 'notes': 'Cyclic encoding'},
        {'feature_name': 'hour_cos', 'data_type': 'float',
         'description': 'Cosine của giờ', 'calculation': 'cos(2π × hour/24)',
         'range': '[-1, 1]', 'notes': 'Cyclic encoding'},
        {'feature_name': 'day_of_week_sin', 'data_type': 'float',
         'description': 'Sine của ngày trong tuần', 'calculation': 'sin(2π × day/7)',
         'range': '[-1, 1]', 'notes': 'Cyclic encoding'},
        {'feature_name': 'day_of_week_cos', 'data_type': 'float',
         'description': 'Cosine của ngày trong tuần', 'calculation': 'cos(2π × day/7)',
         'range': '[-1, 1]', 'notes': 'Cyclic encoding'},
        {'feature_name': 'is_night_transaction', 'data_type': 'int',
         'description': 'Giao dịch ban đêm 23h-5h', 'calculation': 'hour >= 23 or hour <= 5',
         'range': '[0, 1]', 'notes': 'Binary'},
        {'feature_name': 'is_salary_period', 'data_type': 'int',
         'description': 'Kỳ lương ngày 25-5', 'calculation': 'day >= 25 or day <= 5',
         'range': '[0, 1]', 'notes': 'Binary'},

        # Velocity (5)
        {'feature_name': 'tx_count_1h', 'data_type': 'int',
         'description': 'Số giao dịch trong 1 giờ qua', 'calculation': 'rolling count 1H',
         'range': '[0, 100+]', 'notes': 'Dùng shift(1)'},
        {'feature_name': 'tx_count_24h', 'data_type': 'int',
         'description': 'Số giao dịch trong 24 giờ qua', 'calculation': 'rolling count 24H',
         'range': '[0, 100+]', 'notes': 'Dùng shift(1)'},
        {'feature_name': 'minutes_since_last_tx', 'data_type': 'float',
         'description': 'Phút từ giao dịch trước', 'calculation': 'timestamp - prev_timestamp',
         'range': '[0, 10000]', 'notes': 'Clipped at 10000'},
        {'feature_name': 'velocity_change', 'data_type': 'float',
         'description': 'Thay đổi tốc độ giao dịch', 'calculation': 'tx_count_24h / avg_tx_24h_30d',
         'range': '[0, 10]', 'notes': 'Clipped'},
        {'feature_name': 'amount_acceleration', 'data_type': 'float',
         'description': 'Tốc độ tăng số tiền', 'calculation': '(amount - prev) / prev',
         'range': '[-2, 5]', 'notes': 'Clipped'},

        # Recipient (6)
        {'feature_name': 'is_new_recipient', 'data_type': 'int',
         'description': 'Người nhận mới', 'calculation': 'recipient_tx_count == 0',
         'range': '[0, 1]', 'notes': 'Binary'},
        {'feature_name': 'recipient_tx_count', 'data_type': 'int',
         'description': 'Số lần đã giao dịch với người này', 'calculation': 'cumcount theo user-recipient',
         'range': '[0, inf]', 'notes': 'Chỉ tính giao dịch trước'},
        {'feature_name': 'is_same_bank', 'data_type': 'int',
         'description': 'Cùng ngân hàng', 'calculation': 'user_bank == recipient_bank',
         'range': '[0, 1]', 'notes': 'Binary'},
        {'feature_name': 'tx_count_to_same_recipient_24h', 'data_type': 'int',
         'description': 'Số lần chuyển cho người này trong 24h', 'calculation': 'rolling count 24H',
         'range': '[0, 100+]', 'notes': 'Dùng shift(1)'},
        {'feature_name': 'recipient_account_age_days', 'data_type': 'int',
         'description': 'Tuổi tài khoản người nhận', 'calculation': 'timestamp - recipient_created',
         'range': '[0, 3650+]', 'notes': 'Tính tại thời điểm giao dịch'},
        {'feature_name': 'time_since_last_tx_to_recipient', 'data_type': 'float',
         'description': 'Ngày từ lần cuối giao dịch với người này', 'calculation': 'timestamp - prev_tx_to_recipient',
         'range': '[0, 9999]', 'notes': 'Clipped at 9999'},

        # Device/Channel (3)
        {'feature_name': 'is_new_device', 'data_type': 'int',
         'description': 'Device mới', 'calculation': 'device not in user history',
         'range': '[0, 1]', 'notes': 'Binary'},
        {'feature_name': 'channel_encoded', 'data_type': 'int',
         'description': 'Kênh giao dịch', 'calculation': '0:Mobile, 1:Web, 2:ATM, 3:POS',
         'range': '[0, 3]', 'notes': 'Ordinal encoding'},
        {'feature_name': 'is_usual_location', 'data_type': 'int',
         'description': 'Location quen thuộc', 'calculation': 'province in usual_provinces',
         'range': '[0, 1]', 'notes': 'Binary'},

        # Account (2)
        {'feature_name': 'account_age_days', 'data_type': 'int',
         'description': 'Tuổi tài khoản user', 'calculation': 'timestamp - account_created',
         'range': '[0, 3650+]', 'notes': 'Tính tại thời điểm giao dịch'},
        {'feature_name': 'income_level', 'data_type': 'int',
         'description': 'Mức thu nhập', 'calculation': '1:thấp, 2:TB, 3:khá, 4:cao',
         'range': '[1, 4]', 'notes': 'Ordinal'},

        # Scam-specific (3)
        {'feature_name': 'is_incremental_amount', 'data_type': 'int',
         'description': 'Số tiền tăng dần', 'calculation': 'amount > prev_amount (cùng recipient)',
         'range': '[0, 1]', 'notes': 'Binary'},
        {'feature_name': 'is_during_business_hours', 'data_type': 'int',
         'description': 'Trong giờ hành chính', 'calculation': '8h-17h, Mon-Fri',
         'range': '[0, 1]', 'notes': 'Binary'},
        {'feature_name': 'recipient_total_received_24h', 'data_type': 'float',
         'description': 'Tổng tiền người nhận nhận được 24h', 'calculation': 'rolling sum 24H theo recipient',
         'range': '[0, inf]', 'notes': 'Dùng shift(1), từ mọi nguồn'},
    ]

    return pd.DataFrame(dictionary)


def create_metadata(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   test_df: pd.DataFrame, config: dict, scaler: StandardScaler) -> dict:
    """Tạo metadata cho dataset"""

    train_date_range = [str(train_df['timestamp'].min().date()),
                        str(train_df['timestamp'].max().date())]
    val_date_range = [str(val_df['timestamp'].min().date()),
                      str(val_df['timestamp'].max().date())]
    test_date_range = [str(test_df['timestamp'].min().date()),
                       str(test_df['timestamp'].max().date())]

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_transactions": len(df),
        "total_users": config.get('n_users', 0),
        "total_recipients": config.get('n_recipients', 0),
        "date_range": {
            "start": str(df['timestamp'].min().date()),
            "end": str(df['timestamp'].max().date())
        },
        "fraud_rate": config.get('fraud_rate', 0),
        "hack_rate": config.get('hack_ratio', 0.5),
        "scam_rate": config.get('scam_ratio', 0.5),
        "splits": {
            "train": {
                "count": len(train_df),
                "fraud_count": int(train_df['is_fraud'].sum()),
                "date_range": train_date_range
            },
            "validation": {
                "count": len(val_df),
                "fraud_count": int(val_df['is_fraud'].sum()),
                "date_range": val_date_range
            },
            "test": {
                "count": len(test_df),
                "fraud_count": int(test_df['is_fraud'].sum()),
                "date_range": test_date_range
            }
        },
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "random_seed": config.get('seed', 42),
        "scaler_params": {
            "mean": scaler.mean_.tolist(),
            "std": scaler.scale_.tolist(),
            "feature_names": FEATURE_COLUMNS
        }
    }

    return metadata


# ============================================================================
# MAIN DATA GENERATION PIPELINE
# ============================================================================

def generate_full_dataset(n_transactions: int,
                         n_users: int,
                         n_recipients: int,
                         n_months: int,
                         fraud_rate: float,
                         hack_ratio: float,
                         seed: int,
                         progress_bar=None,
                         status_text=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pipeline chính sinh toàn bộ dataset

    Returns:
        users_df, recipients_df, full_data, config
    """
    config = {
        'n_transactions': n_transactions,
        'n_users': n_users,
        'n_recipients': n_recipients,
        'n_months': n_months,
        'fraud_rate': fraud_rate,
        'hack_ratio': hack_ratio,
        'scam_ratio': 1 - hack_ratio,
        'seed': seed
    }

    # Dates
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=n_months * 30)

    def update_status(msg, progress):
        if status_text:
            status_text.text(msg)
        if progress_bar:
            progress_bar.progress(progress)

    # 1. Generate users
    update_status("Đang sinh dữ liệu Users...", 0.1)
    users_df = generate_users(n_users, start_date, seed)

    # 2. Generate recipients
    update_status("Đang sinh dữ liệu Recipients...", 0.2)
    recipients_df = generate_recipients(n_recipients, start_date, seed)

    # 3. Calculate number of fraud transactions
    n_fraud = int(n_transactions * fraud_rate)
    n_hack = int(n_fraud * hack_ratio)
    n_scam = n_fraud - n_hack
    n_normal = n_transactions - n_fraud

    # 4. Generate normal transactions
    update_status(f"Đang sinh {n_normal:,} giao dịch Normal...", 0.3)
    normal_df = generate_normal_transactions(
        users_df, recipients_df, n_normal, start_date, end_date, seed
    )

    # 5. Generate hack transactions
    update_status(f"Đang sinh {n_hack:,} giao dịch Hack/ATO...", 0.5)
    # Mỗi case có 2-5 giao dịch, nên cần n_hack / 3.5 cases
    n_hack_cases = max(1, int(n_hack / 3.5))
    hack_df = generate_hack_transactions(
        users_df, recipients_df, n_hack_cases, start_date, end_date, seed
    )

    # 6. Generate scam transactions
    update_status(f"Đang sinh {n_scam:,} giao dịch Scam...", 0.6)
    # Mỗi case có 2-6 giao dịch
    n_scam_cases = max(1, int(n_scam / 4))
    scam_df = generate_scam_transactions(
        users_df, recipients_df, n_scam_cases, start_date, end_date, seed
    )

    # 7. Combine all
    update_status("Đang kết hợp dữ liệu...", 0.7)
    all_data = pd.concat([normal_df, hack_df, scam_df], ignore_index=True)
    all_data = all_data.sort_values('timestamp').reset_index(drop=True)

    # 8. Assign unique transaction IDs
    all_data['transaction_id'] = [f'TX{i:010d}' for i in range(len(all_data))]

    update_status("Hoàn thành sinh dữ liệu!", 0.8)

    return users_df, recipients_df, all_data, config


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Fraud Detection Data Generator",
        page_icon="🏦",
        layout="wide"
    )

    st.title("🏦 Synthetic Data Generator for Autoencoder Fraud Detection")
    st.markdown("""
    Ứng dụng sinh dữ liệu synthetic chất lượng cao để huấn luyện mô hình Autoencoder
    phát hiện **gian lận (Hack/ATO)** và **lừa đảo (Scam)** trong ngân hàng Việt Nam.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("📊 CẤU HÌNH DỮ LIỆU")

        n_transactions = st.slider(
            "Số lượng giao dịch",
            min_value=10000,
            max_value=500000,
            value=50000,
            step=10000,
            help="Tổng số giao dịch cần sinh"
        )

        n_users = st.slider(
            "Số lượng users",
            min_value=1000,
            max_value=50000,
            value=5000,
            step=1000
        )

        n_recipients = st.slider(
            "Số lượng recipients",
            min_value=5000,
            max_value=100000,
            value=10000,
            step=5000
        )

        n_months = st.selectbox(
            "Khoảng thời gian (tháng)",
            options=[3, 6, 9, 12],
            index=3
        )

        seed = st.number_input(
            "Random seed",
            min_value=1,
            max_value=99999,
            value=42
        )

        st.header("🎯 CẤU HÌNH FRAUD")

        fraud_rate = st.slider(
            "Tỷ lệ fraud tổng",
            min_value=0.01,
            max_value=0.10,
            value=0.03,
            step=0.01,
            format="%.2f",
            help="Tỷ lệ giao dịch gian lận trong tổng số"
        )

        hack_ratio = st.slider(
            "Tỷ lệ Hack vs Scam",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            format="%.1f",
            help="0.5 = 50% Hack + 50% Scam"
        )

        st.header("📁 CHIA DỮ LIỆU")

        train_ratio = st.slider(
            "Train period (%)",
            min_value=50,
            max_value=80,
            value=67,
            help="Phần trăm dữ liệu cho training"
        )

        val_ratio = st.slider(
            "Validation period (%)",
            min_value=10,
            max_value=25,
            value=16
        )

        test_ratio = 100 - train_ratio - val_ratio
        st.info(f"Test period: {test_ratio}%")

        normalize_data = st.checkbox("Export normalized data", value=True)

        st.header("⬇️ EXPORT OPTIONS")

        export_train = st.checkbox("train_normal.parquet", value=True)
        export_val = st.checkbox("validation.parquet", value=True)
        export_test = st.checkbox("test.parquet", value=True)
        export_full = st.checkbox("full_data.parquet", value=True)
        export_scaler = st.checkbox("scaler.pkl", value=True)
        export_dict = st.checkbox("data_dictionary.csv", value=True)
        export_csv = st.checkbox("Also export as CSV", value=False)

    # Main panel
    col1, col2 = st.columns([2, 1])

    with col1:
        generate_btn = st.button("🚀 Sinh Dữ Liệu", type="primary", use_container_width=True)

    with col2:
        st.markdown(f"""
        **Ước tính:**
        - Normal: {int(n_transactions * (1 - fraud_rate)):,}
        - Fraud: {int(n_transactions * fraud_rate):,}
        - Hack: {int(n_transactions * fraud_rate * hack_ratio):,}
        - Scam: {int(n_transactions * fraud_rate * (1 - hack_ratio)):,}
        """)

    if generate_btn:
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Generate data
        users_df, recipients_df, raw_data, config = generate_full_dataset(
            n_transactions=n_transactions,
            n_users=n_users,
            n_recipients=n_recipients,
            n_months=n_months,
            fraud_rate=fraud_rate,
            hack_ratio=hack_ratio,
            seed=seed,
            progress_bar=progress_bar,
            status_text=status_text
        )

        # Compute features
        status_text.text("Đang tính toán 30 features...")
        progress_bar.progress(0.85)

        def feature_progress(progress, msg):
            progress_bar.progress(0.85 + progress * 0.1)
            status_text.text(msg)

        full_data = compute_features(raw_data.copy(), feature_progress)

        # Time-based split
        status_text.text("Đang chia dữ liệu theo thời gian...")
        progress_bar.progress(0.95)

        train_df, val_df, test_df = time_based_split(
            full_data,
            train_ratio / 100,
            val_ratio / 100,
            test_ratio / 100
        )

        # Normalize if requested
        scaler = None
        if normalize_data:
            status_text.text("Đang chuẩn hóa dữ liệu...")
            train_df, val_df, test_df, scaler = fit_and_transform_scaler(
                train_df, val_df, test_df, FEATURE_COLUMNS
            )
        else:
            scaler = StandardScaler()
            scaler.fit(train_df[FEATURE_COLUMNS])

        progress_bar.progress(1.0)
        status_text.text("✅ Hoàn thành!")

        # Store in session state
        st.session_state['train_df'] = train_df
        st.session_state['val_df'] = val_df
        st.session_state['test_df'] = test_df
        st.session_state['full_data'] = full_data
        st.session_state['scaler'] = scaler
        st.session_state['config'] = config
        st.session_state['users_df'] = users_df
        st.session_state['recipients_df'] = recipients_df

    # Display results if data exists
    if 'train_df' in st.session_state:
        train_df = st.session_state['train_df']
        val_df = st.session_state['val_df']
        test_df = st.session_state['test_df']
        full_data = st.session_state['full_data']
        scaler = st.session_state['scaler']
        config = st.session_state['config']

        st.markdown("---")
        st.header("📈 Kết Quả")

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Train (Normal only)", f"{len(train_df):,}")
        with col2:
            st.metric("Validation", f"{len(val_df):,}",
                     f"Fraud: {val_df['is_fraud'].sum():,}")
        with col3:
            st.metric("Test", f"{len(test_df):,}",
                     f"Fraud: {test_df['is_fraud'].sum():,}")
        with col4:
            st.metric("Total", f"{len(full_data):,}",
                     f"Fraud rate: {full_data['is_fraud'].mean():.2%}")

        # Tabs for preview
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Train", "📊 Validation", "📊 Test", "📈 Thống Kê", "🔍 Data Quality"
        ])

        with tab1:
            st.subheader("Train Set (Normal Only)")
            st.dataframe(train_df.head(100), use_container_width=True)

        with tab2:
            st.subheader("Validation Set")
            st.dataframe(val_df.head(100), use_container_width=True)

        with tab3:
            st.subheader("Test Set")
            st.dataframe(test_df.head(100), use_container_width=True)

        with tab4:
            st.subheader("Phân Phối Features")

            # Feature distribution
            feature_to_plot = st.selectbox(
                "Chọn feature để xem phân phối",
                FEATURE_COLUMNS
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Train Set (Normal)**")
                hist_data = train_df[feature_to_plot].dropna()
                st.bar_chart(pd.cut(hist_data, bins=30).value_counts().sort_index())

            with col2:
                st.markdown("**Test Set (Normal vs Fraud)**")
                normal_data = test_df[test_df['is_fraud'] == 0][feature_to_plot].dropna()
                fraud_data = test_df[test_df['is_fraud'] == 1][feature_to_plot].dropna()

                chart_data = pd.DataFrame({
                    'Normal': pd.cut(normal_data, bins=30).value_counts().sort_index().values,
                    'Fraud': pd.cut(fraud_data, bins=30).value_counts().sort_index().values
                })
                st.bar_chart(chart_data)

            # Fraud type distribution
            st.subheader("Phân Phối Loại Fraud")
            fraud_types = pd.concat([val_df, test_df])['fraud_type'].value_counts()
            st.bar_chart(fraud_types)

        with tab5:
            st.subheader("Data Quality Check")

            # Check for nulls
            null_counts = full_data[FEATURE_COLUMNS].isnull().sum()
            if null_counts.sum() == 0:
                st.success("✅ Không có giá trị NULL trong features")
            else:
                st.error(f"❌ Có {null_counts.sum()} giá trị NULL")
                st.dataframe(null_counts[null_counts > 0])

            # Check for variance
            zero_var = full_data[FEATURE_COLUMNS].var() == 0
            if zero_var.sum() == 0:
                st.success("✅ Không có feature nào có variance = 0")
            else:
                st.error(f"❌ Có {zero_var.sum()} features có variance = 0")
                st.write(zero_var[zero_var].index.tolist())

            # Check time ordering
            train_max = train_df['timestamp'].max()
            val_min = val_df['timestamp'].min()
            val_max = val_df['timestamp'].max()
            test_min = test_df['timestamp'].min()

            if train_max <= val_min and val_max <= test_min:
                st.success("✅ Time-based split đúng thứ tự")
            else:
                st.error("❌ Time-based split có vấn đề!")

            # Check train set has no fraud
            train_fraud = train_df['is_fraud'].sum()
            if train_fraud == 0:
                st.success("✅ Train set không chứa fraud")
            else:
                st.error(f"❌ Train set chứa {train_fraud} fraud transactions!")

            st.markdown("---")
            st.markdown("**Feature Statistics**")
            st.dataframe(full_data[FEATURE_COLUMNS].describe().T, use_container_width=True)

        # Export section
        st.markdown("---")
        st.header("⬇️ Export Data")

        col1, col2, col3 = st.columns(3)

        # Create metadata
        metadata = create_metadata(full_data, train_df, val_df, test_df, config, scaler)
        data_dict = create_data_dictionary()

        with col1:
            if export_train:
                buffer = io.BytesIO()
                train_df.to_parquet(buffer, index=False)
                st.download_button(
                    "📥 Download train_normal.parquet",
                    buffer.getvalue(),
                    "train_normal.parquet",
                    "application/octet-stream"
                )

            if export_val:
                buffer = io.BytesIO()
                val_df.to_parquet(buffer, index=False)
                st.download_button(
                    "📥 Download validation.parquet",
                    buffer.getvalue(),
                    "validation.parquet",
                    "application/octet-stream"
                )

        with col2:
            if export_test:
                buffer = io.BytesIO()
                test_df.to_parquet(buffer, index=False)
                st.download_button(
                    "📥 Download test.parquet",
                    buffer.getvalue(),
                    "test.parquet",
                    "application/octet-stream"
                )

            if export_full:
                buffer = io.BytesIO()
                full_data.to_parquet(buffer, index=False)
                st.download_button(
                    "📥 Download full_data.parquet",
                    buffer.getvalue(),
                    "full_data.parquet",
                    "application/octet-stream"
                )

        with col3:
            if export_scaler:
                buffer = io.BytesIO()
                joblib.dump(scaler, buffer)
                st.download_button(
                    "📥 Download scaler.pkl",
                    buffer.getvalue(),
                    "scaler.pkl",
                    "application/octet-stream"
                )

            if export_dict:
                csv_buffer = io.StringIO()
                data_dict.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download data_dictionary.csv",
                    csv_buffer.getvalue(),
                    "data_dictionary.csv",
                    "text/csv"
                )

        # Metadata download
        st.download_button(
            "📥 Download metadata.json",
            json.dumps(metadata, indent=2, ensure_ascii=False),
            "metadata.json",
            "application/json"
        )

        # CSV exports
        if export_csv:
            st.markdown("**CSV Exports:**")
            col1, col2, col3 = st.columns(3)

            with col1:
                csv_buffer = io.StringIO()
                train_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 train_normal.csv",
                    csv_buffer.getvalue(),
                    "train_normal.csv",
                    "text/csv"
                )

            with col2:
                csv_buffer = io.StringIO()
                val_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 validation.csv",
                    csv_buffer.getvalue(),
                    "validation.csv",
                    "text/csv"
                )

            with col3:
                csv_buffer = io.StringIO()
                test_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 test.csv",
                    csv_buffer.getvalue(),
                    "test.csv",
                    "text/csv"
                )

    # Documentation
    with st.expander("📚 Hướng dẫn sử dụng"):
        st.markdown("""
        ### 1. Cấu hình dữ liệu
        - **Số lượng giao dịch**: Tổng số giao dịch cần sinh (khuyến nghị 50,000-200,000)
        - **Số lượng users**: Số lượng người dùng (khuyến nghị 1/10 số giao dịch)
        - **Số lượng recipients**: Số người nhận (khuyến nghị 2x users)

        ### 2. Cấu hình Fraud
        - **Tỷ lệ fraud**: 2-5% là realistic cho ngân hàng VN
        - **Hack vs Scam**: 50/50 hoặc điều chỉnh theo mục đích

        ### 3. Chia dữ liệu
        - **Time-based split**: Bắt buộc để tránh data leakage
        - **Train set**: CHỈ chứa giao dịch normal
        - **Validation/Test**: Chứa cả normal + fraud

        ### 4. Export
        - **Parquet**: Khuyến nghị cho large datasets
        - **Scaler**: Cần thiết để transform data khi inference

        ### 5. Features (30 features)
        Xem chi tiết trong data_dictionary.csv

        ### 6. Checklist tránh Data Leakage
        - ✅ Split theo thời gian, không random
        - ✅ Rolling features dùng shift(1)
        - ✅ Scaler fit chỉ trên train set
        - ✅ Recipient features chỉ dùng data quá khứ
        - ✅ Train set không chứa fraud
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    Fraud Detection Synthetic Data Generator v1.0 |
    Designed for Vietnamese Banking Context |
    Optimized for Autoencoder Training
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
