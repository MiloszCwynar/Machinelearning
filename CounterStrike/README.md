# CS2 AK-47 Price Estimator

A small machine learning notebook that estimates what a Counter-Strike 2 AK-47 skin should cost, based on Steam Market price history.

It takes each skin's latest price and trains a Random Forest to predict price from the skin's **name, wear and StatTrak status**.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib
```

## How to use

Run the cells top to bottom. Then:

**Estimate a skin's price** by editing `new_skin`:

```python
new_skin = pd.DataFrame({
    'skin_name': ['Phantom Disruptor'],
    'wear_condition': ['Factory New'],
    'is_stattrak': [0],   # 0 = normal, 1 = StatTrak
})
```

**Plot any skin's price history** by changing the name in the last cell:

```python
plot_price_history("Desert Eagle | Pilot (Well-Worn)")
```

Names must match the Steam Market name exactly. The "Listing available skins" cell prints every name you have data for.

## How it works

1. `parse_steam_name` splits a market name into skin, wear and StatTrak flag.
2. `load_dataset` reads every AK-47 CSV and keeps the most recent price per skin.
3. Text columns are one-hot encoded, and a Random Forest (200 trees) is trained on `log(price)`.
4. 20% of skins are held back to measure accuracy (MAE and R²).

## Limitations

- It estimates the price **right now**. It does not forecast the future.
- It only knows skins it was trained on. Unseen skin names get a rough guess from wear alone.
- Only AK-47 skins are used. Change `WEAPON` at the top to train on a different weapon.
- With one row per skin, expect a modest R². It is a rough "fair price" estimate, not financial advice.
