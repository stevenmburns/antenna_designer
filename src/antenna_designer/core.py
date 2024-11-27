
def save_or_show(plt, fn):
  if fn is not None:
    if fn != '/dev/null':
      plt.savefig(fn)
  else:
    plt.show()

  plt.close()




