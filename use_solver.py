from solver import create_nng, nng_base

row_res, col_res = create_nng.create_nng_from_image(
    path = 'ressources/Cats.jpg',
    rows=3,
    colums=3,
    show_image=True
)

print(f'row_res = {row_res} ')
print(f'col_res = {col_res}')

nng = nng_base.NNGStupidRec(
    row_restrictions=row_res,
    col_restrictions=col_res,
    save_history=True)
nng.solve()