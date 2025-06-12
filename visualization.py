import matplotlib.pyplot as plt
import io
import logging

logger = logging.getLogger(__name__)

def generate_visualization(df):
    try:
        plt.figure(figsize=(12, 8))
        categories = df['MONATSZAHL'].unique()
        for category in categories:
            cat_data = df[df['MONATSZAHL'] == category]
            yearly_data = cat_data.groupby('JAHR')['WERT'].sum()
            plt.plot(yearly_data.index, yearly_data.values, label=category, linewidth=2.5)

        plt.title('Historical Traffic Accidents in Munich (Until 2020)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Accidents', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        plt.close()

        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        raise