def _to_unicode(str_, py3=True):
    if py3:
        return str_
    else:
        return unicode(str_)
