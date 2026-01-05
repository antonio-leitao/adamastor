extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Attribute, Fields, ItemStruct, Lit, Meta};

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
        let cleaned_field = f.clone();
        let field_name = f.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let field_type = &f.ty;

        // Extract doc comment for description
        let description: Option<String> = f
            .attrs
            .iter()
            .filter_map(|attr| {
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
            })
            .reduce(|acc, s| format!("{} {}", acc, s)); // Combine multiple doc lines

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

        // Generate schema using the type's GeminiSchema implementation
        let field_schema_quote = generate_field_schema(field_type);

        quote! {
            {
                let mut field_schema = #field_schema_quote;
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

/// Generate the schema for a field type, handling nested types properly
fn generate_field_schema(ty: &syn::Type) -> proc_macro2::TokenStream {
    let type_str = quote!(#ty).to_string().replace(" ", "");

    // Handle Vec<T>
    if let Some(inner) = extract_generic_arg(&type_str, "Vec<") {
        let inner_schema = generate_schema_for_type_str(&inner);
        return quote! {
            serde_json::json!({
                "type": "ARRAY",
                "items": #inner_schema
            })
        };
    }

    // Handle Option<T>
    if let Some(inner) = extract_generic_arg(&type_str, "Option<") {
        let inner_schema = generate_schema_for_type_str(&inner);
        return quote! {
            {
                let mut s = #inner_schema;
                if let Some(obj) = s.as_object_mut() {
                    obj.insert("nullable".to_string(), serde_json::json!(true));
                }
                s
            }
        };
    }

    // Handle primitive types and custom types
    generate_schema_for_type_str(&type_str)
}

/// Extract the inner type from a generic like "Vec<Foo>" -> "Foo"
fn extract_generic_arg(type_str: &str, prefix: &str) -> Option<String> {
    if type_str.starts_with(prefix) && type_str.ends_with('>') {
        let inner = &type_str[prefix.len()..type_str.len() - 1];
        Some(inner.to_string())
    } else {
        None
    }
}

/// Generate schema for a type string (primitive or custom)
fn generate_schema_for_type_str(type_str: &str) -> proc_macro2::TokenStream {
    match type_str {
        "String" | "&str" => quote! { serde_json::json!({"type": "STRING"}) },
        "bool" => quote! { serde_json::json!({"type": "BOOLEAN"}) },
        "i32" | "u32" => quote! { serde_json::json!({"type": "INTEGER", "format": "int32"}) },
        "i64" | "u64" => quote! { serde_json::json!({"type": "INTEGER", "format": "int64"}) },
        "f32" => quote! { serde_json::json!({"type": "NUMBER", "format": "float"}) },
        "f64" => quote! { serde_json::json!({"type": "NUMBER", "format": "double"}) },
        _ => {
            // Handle nested Vec/Option that might have been missed
            if let Some(inner) = extract_generic_arg(type_str, "Vec<") {
                let inner_schema = generate_schema_for_type_str(&inner);
                quote! {
                    serde_json::json!({
                        "type": "ARRAY",
                        "items": #inner_schema
                    })
                }
            } else if let Some(inner) = extract_generic_arg(type_str, "Option<") {
                let inner_schema = generate_schema_for_type_str(&inner);
                quote! {
                    {
                        let mut s = #inner_schema;
                        if let Some(obj) = s.as_object_mut() {
                            obj.insert("nullable".to_string(), serde_json::json!(true));
                        }
                        s
                    }
                }
            } else {
                // Custom type - call its GeminiSchema implementation at runtime
                let ty: syn::Type = syn::parse_str(type_str).expect("Failed to parse type");
                quote! { <#ty as adamastor::GeminiSchema>::gemini_schema() }
            }
        }
    }
}
