# R2-score and RMSE, do they change with x-y swap?
# R2-score does. RMSE does not.
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib as mpl

mpl.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)


# to repeat 100 times later

def axis_format(ax):
    ax.set_box_aspect(1)
    ax.set_aspect(1)
    xy_lim = (
        min(
            min(ax.get_xlim()),
            min(ax.get_ylim()),
        ),
        max(
            max(ax.get_xlim()),
            max(ax.get_ylim()),
        ),
    )
    ax.set_xlim(xy_lim)
    ax.set_ylim(xy_lim)
    ax.plot([min(xy_lim), max(xy_lim)],
            [min(xy_lim), max(xy_lim)],
            'k--', linewidth=1)
    return


# make data
x = np.arange(1, 61, 1)[:, np.newaxis]
e = (np.random.randn(60) * 15)[:, np.newaxis]
y_observe = x + e

# make plot of y-x
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
ax.scatter(x, y_observe, c='k')
lr = LinearRegression().fit(x, y_observe)
ax.plot(
    np.array([min(x), max(x)]),
    lr.predict(np.array([min(x), max(x)])),
    'k-', linewidth=2
)
ax.set_xlabel('x')
ax.set_ylabel('y_observe')
axis_format(ax)

# do fitting
lr = LinearRegression().fit(x, y_observe)
y_predict = lr.predict(x)


# make plot of y_fit-y and y_fit-y
def make_plot_of__pred_vs_obs(v_x, v_y, xlabel, ylabel):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.scatter(v_x, v_y, color='k', marker='x')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    axis_format(ax)
    lr = LinearRegression().fit(v_x, v_y)
    v_x_start_end = np.array([np.min(v_x), np.max(v_x)])[:, np.newaxis]
    v_y_start_end = lr.predict(v_x_start_end)
    ax.plot(v_x_start_end, v_y_start_end, 'k-', linewidth=2)
    assert np.isclose(
        (lr.score(v_x, v_y)),
        (np.corrcoef(v_x.flatten(), v_y.flatten())[0, 1] ** 2)
    )
    ax.text(x=10, y=70, s=f"r$^2$={lr.score(v_x, v_y):.2f}")


make_plot_of__pred_vs_obs(v_x=y_observe, v_y=y_predict,
                          xlabel='y_observe', ylabel='y_predict',)
make_plot_of__pred_vs_obs(v_x=y_predict, v_y=y_observe,
                          xlabel='y_predict', ylabel='y_observe',)

