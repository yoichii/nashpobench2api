def get_ith_op(
        cellcode: str,
        i: int
    ):
    # cellcode: 0|12|345
    if i >= 3:
       i += 2
    elif i >= 1:
       i += 1
    return int(cellcode[i])
    
def set_ith_op(
        cellcode: str,
        i: int,
        new_op: int
    ):
    # cellcode: 0|12|345
    if i >= 3:
       i += 2
    elif i >= 1:
       i += 1
    splited = list(cellcode)
    splited[i] = str(new_op)

    return ''.join(splited)


