
def save_or_show(plt, fn):
  if fn is not None:
    plt.savefig(fn)
  else:
    plt.show()

  plt.close()




