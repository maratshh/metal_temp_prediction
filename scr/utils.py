import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def review_data(df: pd.DataFrame, max_unique: int = 10, 
                show_value_counts: bool = False) -> dict:
    """
    Быстрый обзор датасета.
    
    Args:
        df: DataFrame для анализа
        max_unique: Сколько уникальных значений показывать
        show_value_counts: Для категориальных — показывать value_counts вместо списка unique
    
    Returns:
        Словарь с основными метриками
    """
    print(f"{' '*20}Размерность: {df.shape[0]:,} строк x {df.shape[1]} столбцов\n")
    
    # 1. Пропуски
    missing = df.isna().sum()
    missing_percent = (missing / len(df) * 100).round(2)  
    total_missing = missing.sum()
    
    if total_missing > 0:
        print(f"{' '*21}Количество столбцов с пропусками: {df.isna().any().sum()}")
    else:
        print(f"{' '*32}Пропусков нет")
    
    # 2. Дубликаты
    duplicates = df.duplicated().sum()
    print(f"{' '*30}Явные дубликаты: {duplicates:,}")
    
    # 3. Общая информация по колонкам
    print("\nТипы данных и уникальные значения:")
    summary = []
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        n_unique = df[col].nunique()
        missing_pct = missing_percent[col]
        
        print(f"• {col:25} | тип: {col_type:8} | уник: {n_unique:5,} | пропуски: {missing_pct:5.1f}%")
        
        # Дополнительный вывод для категориальных/малых кардинальностей
        if show_value_counts and (df[col].dtype == 'object' or n_unique <= max_unique):
            top_values = df[col].value_counts().head(max_unique)
            if not top_values.empty:
                print("     Топ значений:")
                for val, cnt in top_values.items():
                    print(f"       {val}: {cnt:,}")
        
        summary.append({
            'column': col,
            'dtype': col_type,
            'n_unique': n_unique,
            'missing_pct': missing_pct
        })
    # 4. Статистика числовых столбцов
    numeric_cols = df.select_dtypes(include='number').columns
    
    if len(numeric_cols) > 0:
        print("\nСтатистика числовых столбцов:")
        
        stats = (
            df[numeric_cols]
            .describe()
            .T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        )
        stats.columns = ['mean|', 'std|', 'min|', '[25% ', '50% ', '75%]', 'max|']
        print(stats.to_string(float_format=lambda x: f"{x:,.2f}|"))
        
    print(f"\n{'='*80}")
    
    return df.head(3)

def plot_distribution(data: pd.DataFrame, 
                      column: str, 
                      title: str = None,
                      hue: str = None,
                      figsize: tuple = (9, 3)):
    """
    Строит гистограмму, диаграмму размаха и выводит основные статистики.
    Можно передать hue (категориальный столбец) для раскраски по группам.
    """
    sns.set_style('darkgrid', {"axes.facecolor": ".9"})
    
    if column not in data.columns:
        print(f"Столбец {column} не найден!")
        return
    
    if not pd.api.types.is_numeric_dtype(data[column]):
        print(f"Столбец {column} не числовой!")
        return
    
    if hue is not None:
        if hue not in data.columns:
            print(f"Столбец hue={hue} не найден!")
            return
        if data[hue].nunique() > 10:
            print(f"В hue слишком много уникальных значений ({data[hue].nunique()}). Рекомендуется ≤ 8–10.")
    
    if title is None:
        title = column

    series = data[column].dropna()
    mean_val = series.mean()
    median_val = series.median()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Гистограмма
    bins = np.histogram_bin_edges(data[column].dropna(), bins='doane')
    
    if hue is None:
        sns.histplot(data=data, x=column, bins=bins, kde=True, ax=axes[0])
    else:
        sns.histplot(data=data, x=column, hue=hue, bins=bins, kde=False, 
                     ax=axes[0], element="step", common_norm=False)

    # Вертикальные линии среднего и медианы
    axes[0].axvline(mean_val, color='r', linestyle='--', linewidth=1.5, 
                    label=f'Mean: {mean_val:.2f}')
    axes[0].axvline(median_val, color='g', linestyle='-', linewidth=1.5, 
                    label=f'Median: {median_val:.2f}')
    
    axes[0].set_title('Гистограмма')
    axes[0].set_xlabel(title)
    axes[0].set_ylabel('Частота')
    axes[0].legend(loc='best', fontsize=9)
    
    # Диаграмма размаха
    if hue is None:
        sns.boxplot(data=data, x=column, ax=axes[1])
    else:
        # Для наглядности делаем вертикальные боксплоты по категориям
        sns.boxplot(data=data, y=column, x=hue, ax=axes[1])
        axes[1].set_xlabel(hue)
        axes[1].set_ylabel(title)
    
    axes[1].set_title('Диаграмма размаха')
    
    plt.tight_layout()
    plt.show()

def plot_boxplots(df: pd.DataFrame, 
                  features: list, 
                  title: str = None,
                  figsize: tuple = (9, 4)):
    """Строит диаграммы размаха для заданных признаков на одном графике.

    Аргументы:
        df: pandas DataFrame с данными.
        features: список строк (названий колонок).
        title: заголовок графика.
    """
    # Устанавливаем размер графика в зависимости от числа признаков
    plt.figure(figsize=figsize)

    # Строим график
    sns.boxplot(data=df[features])

    # Настраиваем внешний вид
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Значения", fontsize=12)
    plt.xticks(rotation=45)  # Поворот названий, чтобы не перекрывались
    plt.grid(axis="y", linestyle="--", alpha=0.7)  # Легкая сетка

    plt.tight_layout()
    plt.show()

def plot_scatterplot(
    df,
    features,
    targets=("start_temp", "max_temp", "temp"),
    hue=None,
    height=2,
    s=15
):
    """
    Строит pairplot-подобную матрицу:
    строки — target-признаки,
    столбцы — выбранные features.

    Взаимодействия features между собой и targets между собой
    не отображаются.
    """

    grid = sns.PairGrid(
        df,
        x_vars=features,
        y_vars=targets,
        height=height,
        aspect=1.2
    )

    grid.map(
        sns.scatterplot,
        alpha=0.8,
        s=s
    )

    # Добавляем hue вручную, если нужен is_hot
    if hue is not None:
        for i, target in enumerate(targets):
            for j, feature in enumerate(features):
                ax = grid.axes[i, j]

                sns.scatterplot(
                    data=df,
                    x=feature,
                    y=target,
                    hue=hue,
                    alpha=0.8,
                    s=s,
                    ax=ax,
                    legend=(i == 0 and j == len(features) - 1)
                )

    plt.tight_layout()
    plt.show()

def plot_pairplot(df, diag_kind='hist', corner=False, alpha=0.6, s=15, title=None, hue=None):
    g = sns.pairplot(
        df, corner=corner, hue=hue, diag_kind=diag_kind,
        plot_kws={'alpha': alpha, 's': s})

    g.fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(
    df,
    method='spearman',
    figsize=(16, 12),
    cmap='coolwarm', # 'vlag', 'icefire', 'coolwarm'
    annot=True,
    fmt='.2f',
    title=None,
    mask_upper=False,
    vmin=-1,
    vmax=1
):
    """
    Красивая матрица корреляций.
    
    Parameters
    ----------
    df : pd.DataFrame
    method : str, default 'spearman'
        'pearson', 'spearman' или 'kendall'
    figsize : tuple
    cmap : str
    annot : bool
    fmt : str
    title : str or None
    mask_upper : bool
        Маскировать верхний треугольник (чтобы не дублировать)
    vmin, vmax : float
    """
    
    corr = df.corr(method=method)
    
    # Маска верхнего треугольника
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=figsize)
    
    ax = sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": f"Корреляция ({method})"},
        annot_kws={"size": 9} if annot else None
    )
    
    if title is None:
        title = f"Матрица корреляций ({method.capitalize()})"
    
    plt.title(title, fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()