# Inverted Index with B-Tree Dictionary

سیستم بازیابی اطلاعات با پیاده‌سازی چندین مدل بازیابی و ایندکس معکوس مبتنی بر B-tree

## امکانات

- نرمال‌سازی متن (lowercase، حذف علائم نگارشی)
- توکنایزر ساده
- دیکشنری B-tree برای ذخیره ترم‌ها
- پشتیبانی از PDF
- رابط وب

## سیستم‌های بازیابی

| سیستم | توضیح |
|-------|-------|
| Boolean | عملگرهای AND, OR, NOT |
| TF-IDF | رتبه‌بندی بر اساس فرکانس ترم |
| BM25 | مدل احتمالاتی پیشرفته |
| Probabilistic | مدل استقلال دودویی |

## نمونه کوئری‌ها
```
brain AND tumor
information OR retrieval
deep NOT learning
```

## ساختار پروژه
```
src/
  btree.py              - پیاده‌سازی B-tree
  preprocess.py         - نرمال‌سازی و توکنایز
  inverted_index.py     - ایندکس معکوس
  retrieval_systems.py  - سیستم‌های بازیابی
web/
  static/index.html     - رابط وب
documents/              - محل آپلود فایل‌ها
web_server.py           - سرور Flask
```

## نصب
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## اجرا
```bash
python web_server.py
# http://localhost:5000
```
