extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{Attribute, Fields, ItemStruct, Lit, Meta, parse_macro_input};

#[proc_macro_attribute]
pub fn schema(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(item as ItemStruct);
    let name = &ast.ident;

    ast.vis = syn::parse_quote!(pub);
    let derives_attr: Attribute =
        syn::parse_quote!(#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]);
    ast.attrs.push(derives_attr);

    let (gemini_schema_impl, cleaned_fields) =
        generate_gemini_schema_impl_and_clean_fields(&name, &ast.fields);

    if let Fields::Named(ref mut fields) = ast.fields {
        fields.named = cleaned_fields;
    }

    let output = quote! {
        #ast
        #gemini_schema_impl
    };

    output.into()
}

fn generate_gemini_schema_impl_and_clean_fields(
    name: &syn::Ident,
    fields: &Fields,
) -> (
    proc_macro2::TokenStream,
    syn::punctuated::Punctuated<syn::Field, syn::Token![,]>,
) {
    let fields_iter = match fields {
        Fields::Named(fields) => fields.named.iter(),
        _ => panic!("#[schema] can only be used on structs with named fields"),
    };

    let mut cleaned_fields = syn::punctuated::Punctuated::new();

    let properties_quotes = fields_iter.map(|f| {
        let mut cleaned_field = f.clone();

        let field_name = f.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let field_type = &f.ty;

        let description = f.attrs.iter().find_map(|attr| {
            if attr.path().is_ident("doc") {
                if let Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(expr_lit) = &nv.value {
                        if let Lit::Str(lit_str) = &expr_lit.lit {
                            return Some(lit_str.value().trim().to_string());
                        }
                    }
                }
            }
            None
        });

        cleaned_fields.push(cleaned_field);

        let desc_quote = if let Some(desc) = description {
            quote! {
                if let Some(obj) = field_schema.as_object_mut() {
                    obj.insert("description".to_string(), serde_json::json!(#desc));
                }
            }
        } else {
            quote! {}
        };

        let type_str = quote!(#field_type).to_string().replace(" ", "");

        let field_schema_type =
            if type_str.contains("Vec<String>") || type_str.contains("Vec<&str>") {
                quote! { serde_json::json!({"type": "ARRAY", "items": {"type": "STRING"}}) }
            } else if type_str.contains("Vec<") {
                quote! { serde_json::json!({"type": "ARRAY", "items": {"type": "STRING"}}) }
            } else if type_str.contains("Option<") {
                quote! { serde_json::json!({"type": "STRING", "nullable": true}) }
            } else if type_str.contains("String") {
                quote! { serde_json::json!({"type": "STRING"}) }
            } else if type_str.contains("u32") || type_str.contains("i32") {
                quote! { serde_json::json!({"type": "INTEGER", "format": "int32"}) }
            } else if type_str.contains("u64") || type_str.contains("i64") {
                quote! { serde_json::json!({"type": "INTEGER", "format": "int64"}) }
            } else if type_str.contains("f32") {
                quote! { serde_json::json!({"type": "NUMBER", "format": "float"}) }
            } else if type_str.contains("f64") {
                quote! { serde_json::json!({"type": "NUMBER", "format": "double"}) }
            } else if type_str.contains("bool") {
                quote! { serde_json::json!({"type": "BOOLEAN"}) }
            } else {
                quote! { serde_json::json!({"type": "STRING"}) }
            };

        quote! {
            {
                let mut field_schema = #field_schema_type;
                #desc_quote
                properties.insert(#field_name_str.to_string(), field_schema);
                required.push(#field_name_str.to_string());
            }
        }
    });

    let impl_block = quote! {
        impl adamastor::GeminiSchema for #name {
            fn gemini_schema() -> serde_json::Value {
                let mut properties = serde_json::Map::new();
                let mut required = vec![];

                #(#properties_quotes)*

                serde_json::json!({
                    "type": "OBJECT",
                    "properties": properties,
                    "required": required
                })
            }
        }
    };

    (impl_block, cleaned_fields)
}
